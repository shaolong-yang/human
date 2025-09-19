# human/main.py (示例)
from flask import Flask
# 假设其他 API 路由已经定义
# from human import other_routes
import human.api_inference as api_inference

app = Flask(__name__)

# 注册推理 API 路由
# 注意：实际路由路径可能需要根据你的项目规范调整
@app.route('/api/inference', methods=['POST'])
def inference_endpoint():
    return api_inference.run_inference_api()

# ... 其他路由 ...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # 根据需要调整 host 和 port
