# 企业文档问答 Agent — 私有化部署镜像
# 用法见 DEPLOY.md。核心价值:整个系统跑在客户自己的服务器上,文档与对话不出内网。

FROM python:3.12-slim

WORKDIR /app

# 先单独拷依赖清单并安装,利用 Docker 层缓存:代码改动时不必重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷入应用代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 服务监听端口(容器内),运行时可用 -p 映射到宿主机任意端口
EXPOSE 8000

# 说明:
# - 不配置任何 API Key 时,容器以"纯检索模式"运行,完全离线、不发任何外网请求,
#   适合对数据安全要求极高、连公网大模型都不允许调用的客户。
# - 配置 DEEPSEEK_API_KEY 则启用智能问答;若客户要求绝对不出网,
#   可把 base_url 指向客户内网自建的开源模型(如 vLLM 部署的 Qwen),实现完全本地化。
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
