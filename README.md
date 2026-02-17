## 多尺度用户画像融合的序列推荐系统

### Config

1. 安装依赖

   ```bash
   pip install requirement.txt
   ```

2. 下载Amazon Electronics [Download Link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) ，放入`./data/raw`
3. 下载Embedding模型，`bge-large-en-v1.5`和  LLMs   `LLaMA 3.2 3B -Instruct`

### Run

1. 处理数据

   ```python
   python 0_data_processing.py
   ```

2. 生成语义Embedding

   ```python
   python 1_generate_embedding.py
   ```

3. 生成PCA降维的Embedding

   ```python
   python 2_pca_embedding.py
   ```


4. 生成用户画像和`Embedding`

   ```python
   python 3_generate_profile.py
   ```

5. 运行

   ```python
   python main.py --llm_init --gated
   ```

### Results

|             Model              | NDCG@10 | HR@10  | NDCG@20 | HR@20  |
| :----------------------------: | :-----: | :----: | :-----: | :----: |
|            vanilla             | 0.2124  | 0.3406 | 0.2427  | 0.4609 |
|       vanilla + llm_init       | 0.2488  | 0.3970 | 0.2803  | 0.5224 |
| vanilla+llm_init_gate 0.005 10 | 0.2677  | 0.4313 | 0.3003  | 0.5609 |








