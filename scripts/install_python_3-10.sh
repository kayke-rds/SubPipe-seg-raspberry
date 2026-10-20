#!/usr/bin/env bash
# install_python3-10.sh
# Compila e instala o Python 3.10.12 localmente na pasta do projeto.
# Uso: ./install_python310_local.sh
# Requisitos: base-devel, openssl, zlib, libffi, bzip2, readline, sqlite, xz, tk

set -euo pipefail

PYTHON_VERSION="3.10.12"
PROJECT_ROOT="$(pwd)"
INSTALL_DIR="${PROJECT_ROOT}/python3.10"
BUILD_DIR="${PROJECT_ROOT}/build_python310"

echo "=== Instalando dependências de compilação via pacman ==="
sudo pacman -S --needed --noconfirm \
    base-devel \
    openssl \
    zlib \
    libffi \
    bzip2 \
    readline \
    sqlite \
    xz \
    tk \
    libnsl \
    gdbm \
    mpdecimal \
    wget

echo "=== Baixando Python ${PYTHON_VERSION} ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
tar -xzf "Python-${PYTHON_VERSION}.tgz"
cd "Python-${PYTHON_VERSION}"

echo "=== Configurando compilação com prefixo local ==="
./configure \
    --prefix="${INSTALL_DIR}" \
    --enable-optimizations \
    --with-ensurepip=install \
    --enable-shared \
    LDFLAGS="-Wl,-rpath,${INSTALL_DIR}/lib"

echo "=== Compilando (isso pode levar alguns minutos) ==="
make -j"$(nproc)"

echo "=== Instalando em ${INSTALL_DIR} ==="
make install

echo "=== Limpando diretório de build ==="
cd "${PROJECT_ROOT}"
rm -rf "${BUILD_DIR}"

echo "=== Criando venv com o Python local ==="
"${INSTALL_DIR}/bin/python3.10" -m venv venv_export
source venv_export/bin/activate

echo "=== Atualizando pip e instalando dependências ==="
pip install --upgrade pip setuptools wheel
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
pip install tf-nightly-cpu
pip install ai-edge-torch==0.2.0
pip install opencv-python-headless numpy pillow

echo ""
echo "=== Pronto! ==="
echo "Python 3.10 instalado em: ${INSTALL_DIR}"
echo "Venv criada em: ${PROJECT_ROOT}/venv_export"
echo ""
echo "Para ativar o ambiente em sessões futuras:"
echo "  source venv_export/bin/activate"
echo ""
