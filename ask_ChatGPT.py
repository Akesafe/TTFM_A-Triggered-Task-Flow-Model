import os

'''
To reader:
This script will merge the files in TARGET_FILES into one file (that is, 
ask_ChatGPT_content.txt) and add prompt words to facilitate sending multiple 
files to the AI at one time. You can use it to easily understand, modify and 
review the code.
对读者：
这个脚本会将TARGET_FILES里的文件合并到一个文件里（也就是ask_ChatGPT_content.txt，
并添加提示词，方便一次性发送多个文件给AI。你可以使用它便捷地理解、修改和审阅代码。
'''

# If you need a prompt, you can add it here, else uncomment the line Prompt = "".
# 如果需要提示词，可以一并添加到最后面,如果不需要就把后面那行Prompt = ""取消注释。
Prompt = (
    """
    请简要分析一下这个代码的功能
    """
)

# Prompt = " "

# 待合并的文件列表
TARGET_FILES = [
    #"TTFM_Simulation_main.py",
    #"Experiment_Setup/Model_Generation/Generate_TTFM_Resource.py",
    #"Experiment_Setup/Model_Generation/Convert_STG_to_TTFM_Task.py",
    #"Experiment/exp1_Delay_Composition_Analysis/Exp_dca.py",
    #"Discussion/simulate_for_rho-t.py",
]

OUTPUT_FILES = "ask_ChatGPT_content.txt"  # 实际上更推荐Gemini

# 文件大小警告阈值（单位：字节）
SIZE_LIMIT = 100 * 1024  # 100KB


def merge_files():
    output = []

    for filename in TARGET_FILES:
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"⚠️ File {filename} do not exist, skiped...")
            continue

        # 检查文件大小
        file_size = os.path.getsize(filename)
        if file_size > SIZE_LIMIT:
            choice = input(f"⚠️ The file {filename} more than {file_size // 1024}KB is too large, confirm to conbine? (y/N) ")
            if choice.lower() not in {'y', 'yes'}:
                print(f"skipped {filename}")
                continue

        # 读取文件内容
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                output.append(f"\n------{filename}------\n{content}")
                print(f"ℹ️ {filename} has been combined. ")
        except Exception as e:
            print(f"Cannot open {filename} : {str(e)}")

    # 添加提示词
    if Prompt != "":
        output.append(f"\n------------\n{Prompt}")
        print("ℹ️ Prompt is added")

    # 一次性写入（覆盖模式）
    with open(OUTPUT_FILES, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    print("✅   File combining finished, {} file(s) in total.".format(len(output)))


if __name__ == "__main__":
    merge_files()
