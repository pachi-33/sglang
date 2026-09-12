#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "请使用 sudo 执行此脚本。" >&2
    exit 1
fi

if [[ "$#" -eq 0 ]]; then
    echo "用法：sudo bash $0 镜像[:标签] [其他镜像...]" >&2
    exit 1
fi

command -v docker >/dev/null
command -v systemctl >/dev/null

if ! systemctl is-active --quiet docker.service; then
    echo "docker.service 未运行，请先启动 Docker。" >&2
    exit 1
fi

# 可以从环境变量 PROXY 读取；未设置时隐藏输入。
if [[ -z "${PROXY:-}" ]]; then
    read -r -s -p "请输入代理 URL（http://账号:URL编码密码@IP:端口）：" PROXY
    echo
fi

if [[ -z "$PROXY" ]]; then
    echo "代理地址不能为空。" >&2
    exit 1
fi

# Docker 的 NO_PROXY 域名规则不需要写成 *.huawei.com。
NO_PROXY_VALUE="127.0.0.1,localhost,huawei.com,local,.local,inhuawei.com"

# 转义 systemd Environment= 的特殊字符。
# 特别注意：URL 中的 %40 写入 unit 配置时必须转成 %%40。
systemd_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '%s' "$value"
}

DROPIN_DIR="/run/systemd/system/docker.service.d"
DROPIN_FILE=""

cleanup() {
    local status=$?

    trap - EXIT INT TERM

    if [[ -n "$DROPIN_FILE" ]]; then
        echo "正在删除临时代理配置并恢复 Docker……"
        rm -f -- "$DROPIN_FILE"

        if ! systemctl daemon-reload; then
            echo "警告：daemon-reload 失败，请手动检查。" >&2
            status=1
        fi

        if ! systemctl restart docker.service; then
            echo "警告：Docker 恢复启动失败，请检查 journalctl -u docker。" >&2
            status=1
        fi
    fi

    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$DROPIN_DIR"
DROPIN_FILE="$(mktemp "$DROPIN_DIR/99-pull-proxy-XXXXXX.conf")"
chmod 600 "$DROPIN_FILE"

PROXY_ESCAPED="$(systemd_escape "$PROXY")"
NO_PROXY_ESCAPED="$(systemd_escape "$NO_PROXY_VALUE")"

cat > "$DROPIN_FILE" <<EOF
[Service]
Environment="http_proxy=$PROXY_ESCAPED"
Environment="https_proxy=$PROXY_ESCAPED"
Environment="HTTP_PROXY=$PROXY_ESCAPED"
Environment="HTTPS_PROXY=$PROXY_ESCAPED"
Environment="no_proxy=$NO_PROXY_ESCAPED"
Environment="NO_PROXY=$NO_PROXY_ESCAPED"
EOF

echo "正在应用临时代理配置……"
systemctl daemon-reload
systemctl restart docker.service

for image in "$@"; do
    echo "正在拉取：$image"
    docker pull "$image"
done

echo "镜像拉取完成。"
# 退出时由 cleanup 自动恢复。
