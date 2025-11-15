import os
import re
import jieba
from tqdm import tqdm
import pandas as pd
# 文件路径配置
src_dir = r"D:\python\年报文件\2021\txt年报" # 修改为文本数据的保存地址
tok_dir = r"D:\python\年报文件\2021\分词结果" # 清洗后文本的保存地址
dict_path = r"D:\python\年报文件\环境术语.txt"  # 环境专业词典路径

# 加载环境专业词典
if os.path.exists(dict_path):
    jieba.load_userdict(dict_path)
    print(f"已加载环境专业词典: {dict_path}")
else:
    print(f"词典文件不存在: {dict_path}，将使用默认分词")

# 创建分词结果
for file in tqdm(os.listdir(src_dir)):
    fileRead = os.path.join(src_dir, file)
    fileSave = os.path.join(tok_dir, file + ".txt")
    with open(fileRead, "r", encoding="utf-8") as f:
        text = f.read()  #一次性读入
        for segline in re.split(r"[。；;!?！？…]+", text):  # 按句末标点切段
            #  对段落进行分词处理
            tokens = (w.strip() for w in jieba.cut(segline)) 
            # 将分词后的标点符号替换为空
            re_punc = re.compile(r"[\s+\.\!\/_,$%^*(+\"\'"+"《》【】\[\]{}]+|[+——！，。？、~@#￥%……&*（）：；‘’·]+")
            tokens = [re_punc.sub("", w) for w in tokens if w]   
            # 去掉空字符串
            tokens = [w for w in tokens if w]                    
            with open(fileSave, "a", encoding="utf-8") as g:
                g.write(" ".join(tokens) + "\n") # 用空格将不同的词汇链接起来




# https://radimrehurek.com/gensim/models/word2vec.html # 官方说明文档
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
tok_dir = r"D:\python\年报文件\2021\分词结果" # 调用分词结果
sentences = PathLineSentences(tok_dir)  # 自动遍历目录下所有 分词好的文件
model = Word2Vec(
    vector_size=200,   # 词向量的维度（200维向量表示每个词）
    window=5,          # 上下文窗口大小：看目标词前后5个词作为上下文
    min_count=5,       # 过滤低频词：总频次小于5的词将被忽略，减少噪音和内存
    sg=1,              # 模型训练算法：1=Skip-gram  ；0 = CBOW
    workers=8
)

# 初始化词表（一次遍历）
model.build_vocab(sentences)

# 多轮训练（每轮都会重新从磁盘流式读取）

model.train(sentences, total_examples=model.corpus_count, epochs=1)
model.save(r"D:\python\年报文件\2021\fullModelFordata.model")
Word2Vec.load(r"D:\python\年报文件\2021\fullModelFordata.model")

keywords = ['减排', '可持续发展', '排放', '气候变化', '清洁', '清洁生产', 
           '环保', '环境保护', '生态文明', '监测', '绿色技术', '节能', 
           '节能减排', '低碳']

print("词向量测试：")
for word in keywords:
    if word in model.wv:
        print(f"{word} 的向量形状: {model.wv[word].shape}")
    else:
        print(f"{word} 不在词汇表中")

print("\n相似词测试：")
for word in keywords:
    if word in model.wv:
        print(f"\n与 '{word}' 最相似的词：")
        try:
            similar_words = model.wv.most_similar(word, topn=10)
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.4f}")
        except Exception as e:
            print(f"  错误: {e}")
    else:
        print(f"\n'{word}' 不在词汇表中，无法计算相似词")

# 生成Excel文件
def save_results_to_excel(model, keywords, filename):
    """
    将词向量和相似词结果保存到Excel文件
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        

        
        # 工作表2：相似词
        similar_data = []
        for word in keywords:
            if word in model.wv:
                try:
                    similar_words = model.wv.most_similar(word, topn=10)
                    for rank, (similar_word, score) in enumerate(similar_words, 1):
                        similar_data.append({
                            '关键词': word,
                            '排名': rank,
                            '相似词': similar_word,
                            '相似度': f"{score:.4f}",
                            '相似度数值': round(score, 6)
                        })
                except Exception as e:
                    similar_data.append({
                        '关键词': word,
                        '排名': 'N/A',
                        '相似词': f'错误: {e}',
                        '相似度': 'N/A',
                        '相似度数值': 0
                    })
            else:
                similar_data.append({
                    '关键词': word,
                    '排名': 'N/A',
                    '相似词': '不在词汇表中',
                    '相似度': 'N/A',
                    '相似度数值': 0
                })
        
        df_similar = pd.DataFrame(similar_data)
        df_similar.to_excel(writer, sheet_name='相似词', index=False)
        

    
    print(f"\n结果已保存到: {filename}")

# 调用函数生成Excel
excel_filename = r"D:\python\年报文件\2021\2021.xlsx"
save_results_to_excel(model, keywords, excel_filename)

# 打印一些额外信息
print("\nExcel文件包含以下工作表:")
print("1. 词向量 - 显示每个关键词的词向量（前10维）")
print("2. 相似词 - 显示每个关键词的top10相似词及排名")
print("3. 统计信息 - 显示关键词统计信息和模型信息")
print("\n分析完成！")