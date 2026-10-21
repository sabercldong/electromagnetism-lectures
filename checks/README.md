# 计算复核

保留各讲的独立计算脚本、第一讲和第二三讲的补充核验，以及二轮新增图解关系核验。与原脚本相比，仅将个别读写路径调整为仓库目录。

安装依赖并运行：

```sh
python3 -m pip install -r requirements.txt
python3 scripts/run_checks.py
```

运行结果写入本目录下的 `检查/`、`记录/` 及 JSON 文件；已完成的审查快照见 `audit/calculations/`。部分数值积分脚本需要较长时间。
