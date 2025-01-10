/******************************************************************************
 * 01_basic_syntax.cpp
 *
 * 本文件涵盖 C++ 的基础概念：
 * 1. 基本语法和数据类型
 * 2. 控制流
 * 3. 函数基础
 * 4. 指针和引用
 * 5. 内存管理基础
 *****************************************************************************/

#include <iostream>  // 标准输入输出库
#include <string>    // 字符串库

// 1. 函数声明: 告诉编译器有这样一个函数
int add(int x, int y = 10);   // 提供默认参数 y=10
double add(double x, double y); // 函数重载：函数名一样，但是函数参数不一样（参数个数，参数类型）

int main()
{
    std::cout << "=== 01_basic_syntax.cpp ===" << std::endl << std::endl;

    /**************************************************************************
     * 1. 基本语法和数据类型
     **************************************************************************/
    // 基本数据类型示例
    int a = 1;                 // 整型
    double height = 175.0;     // 双精度浮点型
    char zifu = 'a';           // 字符型
    bool isStudent = true;     // 布尔型

    // 常量 (const)
    const double PI = 3.14;

    // 输出
    std::cout << "[基本数据类型]" << std::endl;
    std::cout << "int a = " << a << std::endl;
    std::cout << "double height = " << height << std::endl;
    std::cout << "char zifu = " << zifu << std::endl;
    std::cout << "bool isStudent = " << isStudent << std::endl;
    std::cout << "const double PI = " << PI << std::endl << std::endl;

    /**************************************************************************
     * 2. 控制流
     *    包含 if / else / else if，for 循环，while 循环等
     **************************************************************************/
    int number = 5;
    std::cout << "[控制流 - if/else/else if]" << std::endl;
    if (number > 0)
    {
        std::cout << number << " is Positive" << std::endl;
    }
    else if (number < 0)
    {
        std::cout << number << " is Negative" << std::endl;
    }
    else
    {
        std::cout << number << " is Zero" << std::endl;
    }
    std::cout << std::endl;

    // for 循环示例
    int count = 0;
    for (int i = 0; i < 5; i++)
    {
        count++;
    }
    std::cout << "[控制流 - for 循环]" << std::endl;
    std::cout << "After for loop, count = " << count << std::endl;

    // while 循环示例
    int i = 0;
    while (i < 5)
    {
        count++;
        i++;
    }
    std::cout << "[控制流 - while 循环]" << std::endl;
    std::cout << "After while loop, count = " << count << std::endl << std::endl;

    /**************************************************************************
     * 3. 函数基础
     *    展示函数的声明、定义、调用，以及函数重载
     **************************************************************************/
    std::cout << "[函数基础]" << std::endl;
    // 调用 add(int x, int y=10)
    int result = add(a, number); 
    std::cout << "add(a, number) = " << result << std::endl;

    // 调用 add(double x, double y)
    double doubleResult = add(3.14, 2.0);
    std::cout << "add(3.14, 2.0) = " << doubleResult << std::endl << std::endl;

    /**************************************************************************
     * 4. 指针和引用
     **************************************************************************/
    std::cout << "[指针和引用]" << std::endl;
    int num = 10;
    int* pNum = &num;   // pNum 指向 num 的地址
    int& rNum = num;    // rNum 是 num 的引用，别名

    // *pNum 表示指针所指向的值
    std::cout << "num = " << num << ", *pNum = " << *pNum 
              << ", rNum = " << rNum << std::endl << std::endl;

    /**************************************************************************
     * 5. 内存管理基础
     *    包含 new/delete、动态数组等
     **************************************************************************/
    std::cout << "[内存管理基础]" << std::endl;
    // 动态分配单个 int
    int* pInt = new int(5); // 分配一个 int，并初始化为 5
    std::cout << "pInt value: " << *pInt << std::endl;
    delete pInt;  // 释放分配的内存

    // 动态分配数组
    int* arr = new int[10];  // 分配了 10 个连续的 int
    arr[0] = 100;
    std::cout << "arr[0] = " << arr[0] << std::endl;
    delete[] arr; // 释放数组内存

    std::cout << std::endl;
    std::cout << "=== 01_basic_syntax.cpp 结束 ===" << std::endl;

    return 0;
}

/******************************************************************************
 * 函数定义
 *****************************************************************************/

// add(int x, int y=10)
int add(int x, int y)
{
    // 演示一下加法
    return x + y;
}

// add(double x, double y) -> 函数重载
double add(double x, double y)
{
    // 演示一下加法 (double 版本)
    return x + y;
}
