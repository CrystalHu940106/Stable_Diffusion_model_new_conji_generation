#!/usr/bin/env python3
"""
Convert Chinese comments to English in all project files
"""

import os
import re
import json
from pathlib import Path

# Translation mapping for common Chinese terms
TRANSLATIONS = {
    # Basic terms
    "诊断": "diagnose",
    "模型": "model", 
    "检查": "check",
    "训练": "training",
    "生成": "generation",
    "修复": "fix",
    "问题": "issue",
    "确保": "ensure",
    "添加": "add",
    "定义": "define", 
    "完成": "complete",
    "测试": "test",
    "使用": "use",
    "创建": "create",
    "扫描": "scan",
    "整个": "entire",
    "项目": "project", 
    "包": "package",
    "能": "can",
    "在": "in",
    "上": "on",
    "运行": "run",
    "的": "of",
    "重点": "key",
    "转换": "convert",
    "相关": "related",
    "脚本": "script",
    "一个": "a",
    "完整": "complete",
    "读取": "read",
    "并": "and",
    "实现": "implementation",
    "代码": "code",
    "启动": "startup",
    "所有": "all",
    "单元格": "cells",
    "开始": "start",
    "开始训练": "start training",
    
    # Technical terms
    "质量": "quality",
    "找出": "find out",
    "黑白色": "black and white",
    "的原因": "cause",
    "权重": "weights",
    "分布": "distribution",
    "范围": "range",
    "标准差": "standard deviation",
    "重建": "reconstruction",
    "能力": "capability",
    "误差": "error",
    "过大": "too large",
    "可能": "may",
    "影响": "affect", 
    "噪声": "noise",
    "预测": "prediction",
    "条件": "condition",
    "效果": "effect",
    "微弱": "weak",
    "不足": "insufficient",
    "全是": "all are",
    "饱和": "saturation",
    "或": "or",
    "初始化": "initialization",
    
    # Method names
    "用不同随机种子测试生成": "test generation with different random seeds",
    "看是否总是黑白色": "check if always black and white",
    "多个": "multiple",
    "随机种子": "random seed",
    "设置": "set",
    "不同": "different",
    "输出": "output",
    "到": "to",
    "控制台": "console",
    "当": "when",
    "执行": "executed",
    "平均值": "mean value",
    "最小值": "minimum value", 
    "最大值": "maximum value",
    "如果": "if",
    "小于": "less than",
    "标准差过小": "standard deviation too small",
    "可能是纯色": "likely solid color",
    "图像": "image",
    
    # Common phrases
    "这样可以": "this can",
    "避免": "avoid",
    "同时": "while",
    "保持": "maintain",
    "在合理范围内": "within reasonable range",
    "而不是": "rather than",
    "软饱和": "soft saturation",
    "更合理的": "more reasonable",
    "调度": "scheduling",
    "替代": "replace",
    "线性调度": "linear scheduling",
    "更": "more",
    "稳定": "stable",
    "系数": "coefficient",
    "版本": "version",
    "使用": "using",
    "温和的": "gentle",
    "激活函数": "activation function",
    "容易饱和在": "easily saturates at",
    "替换": "replace",
    "可学习的": "learnable",
    "缩放因子": "scaling factor",
    "偏移": "offset",
    "解码": "decode",
    "使用可学习的软性激活函数": "use learnable soft activation function",
    "代替硬性": "instead of hard",
    "这样可以避免饱和问题": "this avoids saturation issues",
    "更温和的": "more gentle",
    "更合理的噪声调度": "more reasonable noise scheduling",
    "避免噪声过强": "avoid excessive noise",
    "修复DDPM调度器": "fix DDPM scheduler",
}

def translate_chinese_text(text):
    """Translate Chinese text to English using the mapping"""
    # Sort by length (longest first) to handle multi-character terms first
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    result = text
    for chinese, english in sorted_translations:
        result = result.replace(chinese, english)
    
    return result

def convert_python_file(file_path):
    """Convert Chinese comments in Python files"""
    print(f"Processing Python file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find Chinese characters in comments
    chinese_pattern = r'(#.*?[\u4e00-\u9fff].*?)$'
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # Check if line contains Chinese characters in comments
        if '#' in line and re.search(r'[\u4e00-\u9fff]', line):
            # Extract comment part
            if '#' in line:
                code_part = line[:line.find('#')]
                comment_part = line[line.find('#'):]
                
                # Translate the comment
                translated_comment = translate_chinese_text(comment_part)
                new_line = code_part + translated_comment
                
                if new_line != line:
                    print(f"  Line {i+1}: {line.strip()}")
                    print(f"    -> {new_line.strip()}")
                    lines[i] = new_line
                    modified = True
    
    # Also check for Chinese in docstrings
    docstring_pattern = r'"""(.*?)"""'
    new_content = '\n'.join(lines)
    
    def translate_docstring(match):
        docstring = match.group(1)
        if re.search(r'[\u4e00-\u9fff]', docstring):
            return '"""' + translate_chinese_text(docstring) + '"""'
        return match.group(0)
    
    translated_content = re.sub(docstring_pattern, translate_docstring, new_content, flags=re.DOTALL)
    
    if translated_content != new_content:
        modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ✅ No Chinese comments found in {file_path}")
        return False

def convert_notebook_file(file_path):
    """Convert Chinese comments in Jupyter notebook files"""
    print(f"Processing notebook file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error in {file_path}: {e}")
        print(f"  ⚠️  Skipping malformed notebook file")
        return False
    
    modified = False
    
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = [source]
            
            for line_idx, line in enumerate(source):
                # Check for Chinese characters in comments
                if '#' in line and re.search(r'[\u4e00-\u9fff]', line):
                    # Extract and translate comment
                    if '#' in line:
                        code_part = line[:line.find('#')]
                        comment_part = line[line.find('#'):]
                        translated_comment = translate_chinese_text(comment_part)
                        new_line = code_part + translated_comment
                        
                        if new_line != line:
                            print(f"  Cell {cell_idx}, Line {line_idx}: {line.strip()}")
                            print(f"    -> {new_line.strip()}")
                            source[line_idx] = new_line
                            modified = True
                
                # Also check for Chinese in string literals that might be comments
                if re.search(r'[\u4e00-\u9fff]', line):
                    # Handle print statements and docstrings
                    if 'print(' in line or '"""' in line or "'''" in line:
                        translated_line = translate_chinese_text(line)
                        if translated_line != line:
                            print(f"  Cell {cell_idx}, Line {line_idx}: {line.strip()}")
                            print(f"    -> {translated_line.strip()}")
                            source[line_idx] = translated_line
                            modified = True
            
            cell['source'] = source
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ✅ No Chinese comments found in {file_path}")
        return False

def main():
    """Main function to convert Chinese comments in all files"""
    project_root = Path("/Users/hu.crystal/Documents/NLP/Question2")
    
    print("🔧 Converting Chinese comments to English...")
    print(f"📁 Project root: {project_root}")
    
    total_files = 0
    modified_files = 0
    
    # Process Python files
    for py_file in project_root.rglob("*.py"):
        total_files += 1
        if convert_python_file(py_file):
            modified_files += 1
    
    # Process Jupyter notebook files
    for nb_file in project_root.rglob("*.ipynb"):
        total_files += 1
        if convert_notebook_file(nb_file):
            modified_files += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Files processed: {total_files}")
    print(f"📝 Files modified: {modified_files}")
    print(f"📈 Files unchanged: {total_files - modified_files}")

if __name__ == "__main__":
    main()