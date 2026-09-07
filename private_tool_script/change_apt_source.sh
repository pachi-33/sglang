#!/bin/bash

# 检查是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then
  echo "请使用 sudo 运行此脚本: sudo bash change_source.sh"
  exit 1
fi

echo "正在备份原有的 sources.list..."
cp /etc/apt/sources.list /etc/apt/sources.list.bak

echo "正在写入清华大学 ARM 专用镜像源 (ubuntu-ports)..."
cat << 'EOF' > /etc/apt/sources.list
deb https://mirrors.aliyun.com/ubuntu-ports/ jammy main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu-ports/ jammy main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu-ports/ jammy-security main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu-ports/ jammy-security main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu-ports/ jammy-updates main restricted universe multiverse

# deb https://mirrors.aliyun.com/ubuntu-ports/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu-ports/ jammy-proposed main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu-ports/ jammy-backports main restricted universe multiverse

EOF

echo "正在更新软件列表..."
apt update

echo "换源完成！现在你可以正常安装软件包了，例如: sudo apt install npm"