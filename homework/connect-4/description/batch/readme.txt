1. 新建“Sourcecode”、“so”、“compete_result”目录。
    1）“Sourcecode”目录下以自己的学号建立子目录，存放strategy项目源代码文件,包括Judge.cpp，Judge.h，Strategy.cpp，Strategy.h，Point.h以及自定义的*.cpp和*.h文件，不要包含其他源码文件。
	2）“so”目录下存放自己的so文件，以学号命名。
	3）“TestCases”目录下存放测试样例（注意：所有测试样例需以两位数编号，例如02.so、04.so、06.so、	08.so，100.so以00.so表示）。
2. compile.sh:
	1) 将strategy项目下的源代码编译为可执行so文件。
	2) 需修改root_dir为自己当前目录。
3. compete.sh: 对抗测试
4. stat.py: 统计得分
5. 运行顺序：compile.sh—>compete.sh—>stat.py