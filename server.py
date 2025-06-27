from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import datetime
from feature_extractor import FeatureExtractor

app = Flask(__name__)
app.secret_key = 'your_secret_key'

fe = FeatureExtractor()

# 简单的用户信息存储
users = {'user': '1234'}

# 登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')

# 注册
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            return render_template('register.html', error='用户名已存在')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('register.html')

# 退出登录
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# 主页面（图像上传 + 检索）
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['query_img']
        if file:
            # 读取上传图片
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            query = fe.extract(img_array)

            # 保存上传图片
            uploaded_img_path = "static/image_uploaded/" + datetime.datetime.now().isoformat().replace(":", ".") + "_" + secure_filename(file.filename)
            os.makedirs("static/image_uploaded", exist_ok=True)
            img.save(uploaded_img_path)

            # 加载特征库
            features = np.load("static/feature_database/concat_all_feature.npz")
            paths = features["array_1"]
            feature_vectors = features["array_2"]

            # 计算余弦相似度
            def cosine_similarity(query, X):
                norm_q = np.sqrt(np.sum(query * query))
                norm_X = np.sqrt(np.sum(X * X, axis=1))
                return np.dot(X, query.T).reshape(-1) / (norm_q * norm_X)

            scores = cosine_similarity(query, feature_vectors)
            ids = np.argsort(-scores)[:30]
            scores = scores[ids]
            paths = paths[ids]

            results = zip(paths, scores)
            return render_template("index.html", query_path=uploaded_img_path, results=results)

    return render_template("index.html")
