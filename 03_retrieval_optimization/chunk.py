import os
import re
import json


# =========================
# 1. 配置路径
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "chunks.json")


# =========================
# 2. 读取单份研报
# =========================

def read_report(file_path):
    """
    读取一份 TXT 研报
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


# =========================
# 3. 解析报告元信息
# =========================

def parse_metadata(text):
    """
    从研报开头提取：
    报告编号、标题、分析师、发布日期、投资评级
    """

    metadata = {}

    patterns = {
        "report_id": r"报告编号：(.+)",
        "title": r"报告标题：(.+)",
        "analyst": r"分析师：(.+)",
        "date": r"发布日期：(.+)",
        "rating": r"投资评级：(.+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)

        if match:
            metadata[key] = match.group(1).strip()
        else:
            metadata[key] = ""

    return metadata


# =========================
# 4. 按章节切分
# =========================

def split_into_chunks(text, metadata):
    """
    按一级章节切分。
    
    例如：
    一、行业概况
    二、竞争格局
    三、核心数据
    四、风险提示
    五、投资建议
    
    每个章节作为一个 Chunk。
    """

    # 找到所有一级章节
    pattern = r"(?m)^([一二三四五六七八九十]+、.+)$"

    matches = list(re.finditer(pattern, text))

    chunks = []

    for i, match in enumerate(matches):

        section_title = match.group(1).strip()

        # 当前章节开始位置
        start = match.end()

        # 下一个章节开始位置
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        # 获取章节正文
        section_text = text[start:end].strip()

        # 去掉免责声明
        section_text = re.split(
            r"免责声明：",
            section_text
        )[0].strip()

        # 如果没有正文，就跳过
        if not section_text:
            continue

        # 去掉章节编号，例如：
        # "一、行业概况" -> "行业概况"
        section_name = re.sub(
            r"^[一二三四五六七八九十]+、",
            "",
            section_title
        )

        chunk_id = f"{metadata['report_id']}_{len(chunks) + 1:02d}"

        chunk = {
            "chunk_id": chunk_id,
            "report_id": metadata["report_id"],
            "title": metadata["title"],
            "analyst": metadata["analyst"],
            "date": metadata["date"],
            "rating": metadata["rating"],
            "section": section_name,
            "text": section_text
        }

        chunks.append(chunk)

    return chunks


# =========================
# 5. 处理全部研报
# =========================

def process_all_reports():

    all_chunks = []

    # 获取所有 TXT 文件
    files = [
        file_name
        for file_name in os.listdir(REPORT_DIR)
        if file_name.lower().endswith(".txt")
    ]

    # 排序，方便结果稳定
    files.sort()

    print(f"发现 {len(files)} 份研报")

    for file_name in files:

        file_path = os.path.join(
            REPORT_DIR,
            file_name
        )

        print(f"正在处理：{file_name}")

        # 读取
        text = read_report(file_path)

        # 元信息
        metadata = parse_metadata(text)

        # Chunk
        chunks = split_into_chunks(
            text,
            metadata
        )

        print(
            f"  → {metadata['report_id']} "
            f"切分得到 {len(chunks)} 个 chunks"
        )

        all_chunks.extend(chunks)

    # 保存
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n处理完成！")
    print(f"总 Chunk 数量：{len(all_chunks)}")
    print(f"输出文件：{OUTPUT_FILE}")


# =========================
# 6. 程序入口
# =========================

if __name__ == "__main__":
    process_all_reports()