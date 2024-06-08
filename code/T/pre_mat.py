from scipy.io import loadmat

# 替换为你的.mat文件路径
mat_file_path = 'C://Users//caifengze//Desktop//CPDS_PATENE//data//00_25J_RVI_38_Full//RVI_38_labels.mat'

# mat_file_path = 'C:\\Users\\caifengze\\Desktop\\CPDS_PATENE\\data\\labels.mat'
# 加载.mat文件
data = loadmat(mat_file_path)

print(data)
# data是一个字典，包含了文件中所有的变量
# 你可以通过变量名作为键来访问这些变量的值
