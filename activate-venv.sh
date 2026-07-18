#!/usr/bin/env bash

# Ativa o ambiente virtual no shell atual.
# Uso: source ./activate-venv.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Erro: ambiente virtual não encontrado em $VENV_DIR" >&2
  echo "Crie-o com: python3 -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"
echo "venv ativado: $VENV_DIR"
