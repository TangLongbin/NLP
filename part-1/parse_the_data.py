import json
import os


# 数据读写文件路径
DataPath = "data/document_parses"
OutTemplateFile = "parsed_data_texts/text"



# 递归遍历一个字典的所有键值对
def dfs_dict(dict_now, OutPutFile):
    
    if not isinstance(dict_now, dict):
        return
    
    keys = dict_now.keys()
    for key in keys:
        # 读取文件中所有“text”的内容（可根据需求替换）
        if key == "text" and len(dict_now[key]) > 20:
            OutPutFile.write(dict_now[key] + '\n')
        else:
            dfs_dict(dict_now[key], OutPutFile)
    
    return


# 获取指定文件夹内的所有文件的绝对路径
# return type: List
def GetFilePath(DirPath):
    
    Res = []
    for FilePath, DirNames, FileNames in os.walk(DirPath):
        for FileName in FileNames:
            str_tmp = os.path.join(FilePath, FileName)
            Res.append(str_tmp)
            
    return Res


# 获取指定文件的“值”内容
def GetValueFromFile(FilePath, OutPutFile):
    
    DataFile = open(FilePath, "r")
    DictTmp = json.load(DataFile)
    dfs_dict(DictTmp, OutPutFile)
    DataFile.close()
    
    return
    
    
def main():
    
    # 获取所有文件路径并依次遍历
    FilePaths = GetFilePath(DataPath)
    cnt = 0
    for FilePath in FilePaths:
        OutPutPath = OutTemplateFile + "_" + str(cnt) + ".txt"
        OutPutFile = open(OutPutPath, "w")
        GetValueFromFile(FilePath, OutPutFile)
        print("\r",end="")
        print(cnt, end="")
        cnt += 1
        OutPutFile.close()
    
    return
        

if __name__ == "__main__":
    main()

    