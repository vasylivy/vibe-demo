# Demo Development Container
FROM ubuntu:24.04

# 1. System dependencies
# - clang, lld, libc++-dev: LLVM C/C++ compiler, linker, and standard library
# - openmpi-bin, libopenmpi-dev: MPI runtime + headers for parallel computing
# - cmake, ninja-build: build systems for C/C++ projects
# - git: needed by Claude Code CLI for git operations
# - tmux: needed by Claude Code agent teams (--teammate-mode tmux)
# - openssh-server: SSH access for iTerm2 tmux -CC integration
# - ncurses-term, locales: 256-color terminal + UTF-8 support
# - python3, pip: scripting, pre/post-processing, plotting
# - libboost-dev: STK transitive header dependency
# - libnetcdf-*, libhdf5-openmpi-dev: SEACAS/Exodus I/O
# - libparmetis-dev, libmetis-dev: Zoltan2 partitioning
# - libsuitesparse-dev: KLU2 sparse direct solver (MueLu coarse grid)
# - libblas-dev, liblapack-dev: Teuchos/Tpetra dense kernels
RUN apt-get update && apt-get install -y \
    clang \
    clang-tidy \
    clang-format \
    clangd-18 \
    lld \
    libc++-dev \
    libc++abi-dev \
    openmpi-bin \
    libopenmpi-dev \
    cmake \
    ninja-build \
    git \
    curl \
    tmux \
    gh \
    gosu \
    openssh-server \
    ncurses-term \
    locales \
    jq \
    lcov \
    libomp-18-dev \
    libhdf5-openmpi-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libnetcdf-dev \
    libnetcdf-c++4-dev \
    libparmetis-dev \
    libmetis-dev \
    libsuitesparse-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    python3 \
    python3-pip \
    python3-venv \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/clangd-18 /usr/bin/clangd

# Python packages for visualization (h5py, numpy, matplotlib, scipy)
RUN pip3 install --break-system-packages numpy matplotlib h5py scipy

# Generate en_US.UTF-8 locale for proper Unicode rendering
RUN sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen

# Create non-root user (UID/GID configurable at build time)
# Ubuntu 24.04 images ship a default 'ubuntu' user at UID/GID 1000 — remove it first
ARG UID=1000
ARG GID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (groupdel ubuntu 2>/dev/null || true) \
    && groupadd -g ${GID} demo \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash demo \
    && mkdir -p /home/demo/.ssh && chmod 700 /home/demo/.ssh \
    && chown -R demo:demo /home/demo

# Configure SSH: key-only auth, no passwords, no root login
RUN mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Terminal + locale environment
ENV TERM=xterm-256color \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    COLORTERM=truecolor

# 2. Default mpicxx / mpicc to clang so -stdlib=libc++ works everywhere.
#    Trilinos itself is built on demand inside the container via
#    scripts/build_trilinos.sh — not baked into the image.
ENV OMPI_CC=clang OMPI_CXX=clang++

# 3. Install Node.js (required for Claude Code + Gemini CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Claude Code + Gemini CLI globally
RUN npm install -g @anthropic-ai/claude-code @google/gemini-cli

# 5. Working directory
WORKDIR /demo

# 6. Entrypoint — runs AFTER volumes are mounted, BEFORE the CMD
COPY docker-entrypoint.sh ./
ENTRYPOINT ["./docker-entrypoint.sh"]

# 7. Expose SSH port (add solver ports as needed)
EXPOSE 22

# 8. Default command — interactive shell
CMD ["bash"]
