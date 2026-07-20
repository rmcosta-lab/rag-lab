from __future__ import annotations

from typing import Any, Dict, List
import os
import signal
import subprocess
import sys
import time

import psutil


def kill_processes_on_ports(
    ports: List[int],
    *,
    only_listening: bool = True,
    include_udp: bool = True,
    force: bool = True,
    timeout: float = 5.0,
    protect_current_process: bool = True,
) -> Dict[str, Any]:
    """
    Encerra processos vinculados às portas informadas.

    No macOS, utiliza `lsof`, pois psutil.net_connections() exige privilégios
    elevados para enumerar conexões de todos os processos.

    Args:
        ports:
            Portas que serão verificadas.
        only_listening:
            Para TCP, considera somente processos em estado LISTEN.
        include_udp:
            Inclui processos vinculados a sockets UDP.
        force:
            Usa SIGKILL caso SIGTERM não encerre o processo no prazo.
        timeout:
            Tempo máximo, em segundos, para aguardar após SIGTERM.
        protect_current_process:
            Impede que o processo Python atual e seus processos-pai sejam mortos.

    Returns:
        Dicionário com PIDs encontrados, encerrados e eventuais erros.
    """

    target_ports = sorted({int(port) for port in ports})

    for port in target_ports:
        if not 1 <= port <= 65535:
            raise ValueError(f"Porta inválida: {port}")

    results: Dict[str, Any] = {
        "pids_targeted": [],
        "terminated": [],
        "killed": [],
        "protected": [],
        "errors": [],
        "ports_with_no_match": [],
        "port_pid_map": {},
    }

    protected_pids: set[int] = set()

    if protect_current_process:
        current = psutil.Process(os.getpid())

        while current is not None:
            protected_pids.add(current.pid)

            try:
                current = current.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    pids: set[int] = set()
    matched_ports: set[int] = set()

    for port in target_ports:
        port_pids: set[int] = set()

        # TCP
        tcp_command = [
            "lsof",
            "-nP",
            "-t",
            f"-iTCP:{port}",
        ]

        if only_listening:
            tcp_command.append("-sTCP:LISTEN")

        port_pids.update(_run_lsof(tcp_command, port, results))

        # UDP
        if include_udp:
            udp_command = [
                "lsof",
                "-nP",
                "-t",
                f"-iUDP:{port}",
            ]

            port_pids.update(_run_lsof(udp_command, port, results))

        if port_pids:
            matched_ports.add(port)
            pids.update(port_pids)

        results["port_pid_map"][port] = sorted(port_pids)

    results["ports_with_no_match"] = sorted(
        set(target_ports) - matched_ports
    )

    # Nunca encerra o próprio kernel/notebook nem seus processos-pai.
    safe_pids = pids - protected_pids
    blocked_pids = pids & protected_pids

    for pid in sorted(blocked_pids):
        results["protected"].append(
            {
                "pid": pid,
                "name": _safe_process_name(pid),
                "reason": "Current Python process or parent process",
            }
        )

    results["pids_targeted"] = sorted(safe_pids)

    processes: list[psutil.Process] = []

    for pid in sorted(safe_pids):
        try:
            process = psutil.Process(pid)

            if not process.is_running():
                continue

            process_name = _safe_process_name(pid)

            process.terminate()
            processes.append(process)

            # Guardamos o nome antes que o processo deixe de existir.
            process._cached_display_name = process_name  # type: ignore[attr-defined]

        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue

        except psutil.AccessDenied as exc:
            results["errors"].append(
                {
                    "pid": pid,
                    "error": f"Access denied on terminate: {exc}",
                }
            )

        except Exception as exc:
            results["errors"].append(
                {
                    "pid": pid,
                    "error": str(exc),
                }
            )

    gone, alive = psutil.wait_procs(processes, timeout=timeout)

    for process in gone:
        results["terminated"].append(
            {
                "pid": process.pid,
                "name": getattr(
                    process,
                    "_cached_display_name",
                    _safe_process_name(process.pid),
                ),
            }
        )

    if alive and force:
        for process in alive:
            try:
                process.kill()

            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

            except psutil.AccessDenied as exc:
                results["errors"].append(
                    {
                        "pid": process.pid,
                        "error": f"Access denied on kill: {exc}",
                    }
                )

            except Exception as exc:
                results["errors"].append(
                    {
                        "pid": process.pid,
                        "error": str(exc),
                    }
                )

        gone_after_kill, still_alive = psutil.wait_procs(
            alive,
            timeout=timeout,
        )

        for process in gone_after_kill:
            results["killed"].append(
                {
                    "pid": process.pid,
                    "name": getattr(
                        process,
                        "_cached_display_name",
                        _safe_process_name(process.pid),
                    ),
                }
            )

        for process in still_alive:
            results["errors"].append(
                {
                    "pid": process.pid,
                    "error": "Process is still alive after kill()",
                }
            )

    return results


def _run_lsof(
    command: List[str],
    port: int,
    results: Dict[str, Any],
) -> set[int]:
    """Executa lsof e devolve os PIDs encontrados."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    except FileNotFoundError:
        raise RuntimeError(
            "`lsof` não foi encontrado. No macOS ele normalmente já vem "
            "instalado no sistema."
        )

    # lsof retorna 1 quando nenhuma conexão é encontrada.
    if completed.returncode not in (0, 1):
        results["errors"].append(
            {
                "port": port,
                "error": completed.stderr.strip()
                or f"lsof returned code {completed.returncode}",
            }
        )
        return set()

    pids: set[int] = set()

    for line in completed.stdout.splitlines():
        line = line.strip()

        if line.isdigit():
            pids.add(int(line))

    return pids


def _safe_process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
    ):
        return "?"