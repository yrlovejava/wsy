import os
import re
import jieba
from tqdm import tqdm
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


def load_environment_dictionary(dict_path):
    """加载环境专业词典"""
    if os.path.exists(dict_path):
        jieba.load_userdict(dict_path)
        print(f"已加载环境专业词典: {dict_path}")
    else:
        print(f"词典文件不存在: {dict_path}，将使用默认分词")


def preprocess_text(text):
    """文本预处理：分句、分词、清洗"""
    processed_sentences = []
    
    # 按句末标点切分段落
    for segline in re.split(r"[。；;!?！？…]+", text):
        # 对段落进行分词处理
        tokens = (w.strip() for w in jieba.cut(segline))
        
        # 清洗分词结果：去除标点符号和空字符
        re_punc = re.compile(r"[\s+\.\!\/_,$%^*(+\"\'" + "《》【】\[\]{}]+|[+——！，。？、~@#￥%……&*（）：；‘’·]+")
        tokens = [re_punc.sub("", w) for w in tokens if w]
        tokens = [w for w in tokens if w]
        
        if tokens:  # 只添加非空句子
            processed_sentences.append(" ".join(tokens))
    
    return processed_sentences


def process_corpus_files(src_dir, tok_dir):
    """处理语料库文件：读取、分词、保存"""
    print("开始处理语料库文件...")
    
    for file in tqdm(os.listdir(src_dir)):
        file_read = os.path.join(src_dir, file)
        file_save = os.path.join(tok_dir, file + ".txt")
        
        with open(file_read, "r", encoding="utf-8") as f:
            text = f.read()  # 一次性读入
            
        # 文本预处理
        processed_sentences = preprocess_text(text)
        
        # 保存处理后的文本
        with open(file_save, "w", encoding="utf-8") as g:
            for sentence in processed_sentences:
                g.write(sentence + "\n")


def train_word2vec_model(tok_dir, model_save_path):
    """训练Word2Vec模型"""
    print("开始训练Word2Vec模型...")
    
    # 自动遍历目录下所有分词好的文件
    sentences = PathLineSentences(tok_dir)
    
    # 初始化模型参数
    model = Word2Vec(
        vector_size=200,   # 词向量的维度（200维向量表示每个词）
        window=5,          # 上下文窗口大小：看目标词前后5个词作为上下文
        min_count=5,       # 过滤低频词：总频次小于5的词将被忽略
        sg=1,              # 模型训练算法：1=Skip-gram；0=CBOW
        workers=8          # 并行工作线程数
    )
    
    # 构建词表并训练模型
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=1)
    
    # 保存模型
    model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    return model


def test_model_keywords(model, keywords):
    """测试模型关键词"""
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


def save_results_to_excel(model, keywords, filename):
    """
    将词向量和相似词结果保存到Excel文件
    """
    print("开始生成Excel结果文件...")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 工作表1：相似词
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
    
    print(f"结果已保存到: {filename}")


def main():
    """主函数"""
    # 文件路径配置
    src_dir = r"D:\python\年报文件\2021\txt年报"  # 修改为文本数据的保存地址
    tok_dir = r"D:\python\年报文件\2021\分词结果"  # 清洗后文本的保存地址
    dict_path = r"D:\python\年报文件\环境术语.txt"  # 环境专业词典路径
    model_save_path = r"D:\python\年报文件\2021\fullModelFordata.model"
    excel_filename = r"D:\python\年报文件\2021\2021.xlsx"
    
    # 关键词列表
    keywords = [
        '减排', '可持续发展', '排放', '气候变化', '清洁', '清洁生产', 
        '环保', '环境保护', '生态文明', '监测', '绿色技术', '节能', 
        '节能减排', '低碳'
    ]
    
    # 确保输出目录存在
    os.makedirs(tok_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(excel_filename), exist_ok=True)
    
    # 1. 加载环境专业词典
    load_environment_dictionary(dict_path)
    
    # 2. 处理语料库文件
    process_corpus_files(src_dir, tok_dir)
    
    # 3. 训练Word2Vec模型
    model = train_word2vec_model(tok_dir, model_save_path)
    
    # 4. 加载模型并测试关键词
    model = Word2Vec.load(model_save_path)
    test_model_keywords(model, keywords)
    
    # 5. 生成Excel结果文件
    save_results_to_excel(model, keywords, excel_filename)
    
    # 6. 打印总结信息
    print("\nExcel文件包含以下工作表:")
    print("1. 相似词 - 显示每个关键词的top10相似词及排名")
    print("\n分析完成！")


if __name__ == "__main__":
    main()