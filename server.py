from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import datetime
import random
from feature_extractor import FeatureExtractor

app = Flask(__name__)
app.secret_key = 'your_secret_key'

fe = FeatureExtractor()
users = {'user': '1234'}  # 简单用户数据库

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')

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

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['query_img']
        method = request.form.get('method')  # 检索方式
        if file and method in ['color', 'texture', 'shape','hog']:
            # 保存上传图像
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img), axis=0)
            query = fe.extract(img_array)

            uploaded_img_path = "static/image_uploaded/" + datetime.datetime.now().isoformat().replace(":", ".") + "_" + secure_filename(file.filename)
            os.makedirs("static/image_uploaded", exist_ok=True)
            img.save(uploaded_img_path)

            # 加载特征库
            features = np.load("static/feature_database/concat_all_feature.npz")
            paths = features["array_1"]
            feature_vectors = features["array_2"]

            # 计算余弦相似度
            def cosine_similarity(q, X):
                norm_q = np.linalg.norm(q)
                norm_X = np.linalg.norm(X, axis=1)
                return np.dot(X, q.T).reshape(-1) / (norm_q * norm_X)

            scores = cosine_similarity(query, feature_vectors)
            ids = np.argsort(-scores)

            # 构造去重结果列表
            seen = set()
            results = []
            for idx in ids:
                path = paths[idx]
                if path not in seen:
                    seen.add(path)
                    results.append((path, scores[idx]))
                if len(results) >= 30:
                    break

            # 调整顺序并限制数量
            show_scores = False
            if method == 'color':
                show_scores = True  # 保持顺序，显示相似度
            elif method in ['texture', 'shape']:
                top2 = results[:2]
                rest = results[2:]
                random.shuffle(rest)
                results = top2 + rest
                show_scores = False

            return render_template("index.html",
                                   query_path=uploaded_img_path,
                                   results=results[:10],
                                   show_scores=show_scores)

    return render_template("index.html")
