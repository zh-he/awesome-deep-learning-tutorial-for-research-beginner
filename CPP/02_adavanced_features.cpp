/******************************************************************************
 * 02_advanced_features.cpp
 *
 * 本文件涵盖 C++ 的进阶特性：
 * 1. 面向对象编程 (OOP)
 * 2. STL 容器和算法
 * 3. Lambda 表达式和现代 C++ 特性
 * 4. 模板和泛型编程
 * 5. 多线程编程基础
 *
 *****************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <stdexcept>
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::lock_guard
#include <condition_variable> // std::condition_variable
#include <future>             // std::async, std::future
#include <chrono>             // std::chrono::seconds, std::this_thread::sleep_for

/******************************************************************************
 * 第 1 部分：面向对象编程 (OOP)
 *
 * 核心概念：
 * 1. 类与对象：类是对象的模板，对象是类的实例
 * 2. 封装：通过访问修饰符（private/protected/public）控制成员的可见性
 * 3. 继承：子类继承父类的特性，实现代码重用
 * 4. 多态：通过虚函数 (virtual) 实现运行时的动态绑定
 * 5. 构造函数和析构函数：管理对象的生命周期
 *    - 构造函数：对象创建时自动执行，用于初始化成员变量或分配资源
 *    - 析构函数：对象销毁时自动执行，用于释放资源或做清理操作
 *****************************************************************************/

// 定义一个基类 Person
class Person
{
public:
    // 构造函数：在创建对象时自动调用，可在此进行成员变量初始化
    Person(const std::string &name, int age) : name_(name), age_(age)
    {
        std::cout << "[Person 构造函数] name = " << name_
                  << ", age = " << age_ << std::endl;
    }

    // 虚析构函数：多态场景下，确保派生类的析构函数也能被正确调用
    virtual ~Person()
    {
        std::cout << "[Person 析构函数]" << std::endl;
    }

    // 公共接口：打印对象信息
    void printInfo() const
    {
        std::cout << "Name: " << name_
                  << ", Age: " << age_ << std::endl;
    }

    // 虚函数：允许派生类覆盖，体现多态
    virtual void testVirtual()
    {
        std::cout << "I am a Person" << std::endl;
    }

private:
    std::string name_; // 封装的数据成员
    int age_;
};

// 派生类 Teenager，演示继承和多态
class Teenager : public Person
{
public:
    // 构造函数：利用初始化列表调用基类的构造函数
    Teenager(const std::string &name, int age)
        : Person(name, age)
    {
        std::cout << "[Teenager 构造函数]" << std::endl;
    }

    // 覆盖基类的虚析构函数
    ~Teenager() override
    {
        std::cout << "[Teenager 析构函数]" << std::endl;
    }

    // 重写虚函数，实现多态
    void testVirtual() override
    {
        std::cout << "I am a Teenager" << std::endl;
    }
};

/******************************************************************************
 * 第 2 部分：STL (Standard Template Library)
 *
 * 核心概念：
 * 1. 容器：存储和组织数据的数据结构
 *    - 序列容器：vector, list, deque
 *    - 关联容器：set, map
 *    - 无序容器：unordered_set, unordered_map
 * 2. 迭代器：用于遍历容器的接口
 * 3. 算法：独立于容器的通用算法（如排序、查找）
 *
 * 下面演示了部分容器的 CRUD 操作（Create, Read, Update, Delete）。
 *****************************************************************************/

void demonstrateSTL()
{
    std::cout << "[STL 容器与算法演示 - CRUD 示例]" << std::endl;

    // 1. vector：动态数组
    //  (1) Create
    std::vector<int> vec = {1, 2, 3};
    //  (2) Read
    std::cout << "vector 初始化内容: ";
    for (auto &val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    //  (3) Update
    vec.push_back(4); // 在末尾添加元素
    vec[0] = 10;      // 修改第一个元素
    //  (4) Delete
    vec.pop_back(); // 删除末尾元素

    std::cout << "vector 修改后内容: ";
    for (auto &val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl
              << std::endl;

    // 2. map：键值对容器
    //  (1) Create
    std::map<std::string, int> ages;
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    //  (2) Read
    std::cout << "map 内容: ";
    for (auto &kv : ages)
    {
        std::cout << kv.first << ":" << kv.second << " ";
    }
    std::cout << std::endl;
    //  (3) Update
    ages["Alice"] = 26; // 更新 Alice 的年龄
    //  (4) Delete
    ages.erase("Bob"); // 删除 Bob 这条记录

    std::cout << "map 修改后内容: ";
    for (auto &kv : ages)
    {
        std::cout << kv.first << ":" << kv.second << " ";
    }
    std::cout << std::endl
              << std::endl;

    // 3. set：唯一元素集合
    //  (1) Create
    std::set<int> numSet = {3, 1, 4, 1, 5}; // 重复元素不会插入
    //  (2) Read
    std::cout << "set 内容: ";
    for (auto &val : numSet)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    //  (3) Update
    // set 中没有直接“更新”已有元素的操作，一般先删后插
    if (numSet.find(3) != numSet.end())
    {
        numSet.erase(3);
        numSet.insert(33); // 相当于“更新”
    }
    //  (4) Delete
    numSet.erase(1);

    std::cout << "set 修改后内容: ";
    for (auto &val : numSet)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl
              << std::endl;

    // 4. list：双向链表容器
    //  (1) Create
    std::list<int> myList = {10, 20, 30};
    //  (2) Read
    std::cout << "list 内容: ";
    for (auto &val : myList)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    //  (3) Update
    myList.push_front(5);     // 头部插入
    *(++myList.begin()) = 15; // 修改第二个元素
    //  (4) Delete
    myList.pop_back(); // 删除尾部元素

    std::cout << "list 修改后内容: ";
    for (auto &val : myList)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl
              << std::endl;

    // 常用算法演示：sort / find
    std::vector<int> algoVec = {4, 1, 3, 2, 5};
    std::sort(algoVec.begin(), algoVec.end());              // 排序
    auto it = std::find(algoVec.begin(), algoVec.end(), 3); // 查找
    if (it != algoVec.end())
    {
        std::cout << "在 algoVec 中找到了 3" << std::endl;
    }

    std::cout << "algoVec 排序后: ";
    for (auto &val : algoVec)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl
              << std::endl;
}

/******************************************************************************
 * 第 3 部分：Lambda 表达式和现代 C++ 特性
 *
 * 核心概念：
 * 1. Lambda 表达式：创建匿名函数对象
 *    语法：[捕获列表](参数列表) -> 返回类型 { 函数体 }
 * 2. auto：类型推导
 * 3. 范围 for 循环：简化容器遍历
 * 4. 智能指针：自动内存管理（如 std::unique_ptr, std::shared_ptr）
 *****************************************************************************/

void demonstrateModernCpp()
{
    std::cout << "[Lambda 表达式与现代 C++ 特性]" << std::endl;

    // 1. 简单 Lambda 示例
    auto add = [](int a, int b)
    {
        return a + b;
    };
    int sum = add(3, 5);
    std::cout << "add(3, 5) = " << sum << std::endl;

    // 2. 带捕获的 Lambda
    int multiplier = 10;
    auto multiply = [multiplier](int x)
    {
        return x * multiplier;
    };
    std::cout << "multiply(4) = " << multiply(4) << std::endl;

    // 3. 范围 for 循环
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "numbers: ";
    for (const auto &num : numbers)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl
              << std::endl;
}

/******************************************************************************
 * 第 4 部分：模板与泛型编程
 *
 * 核心概念：
 * 1. 函数模板：生成适用于不同类型的函数
 * 2. 类模板：生成适用于不同类型的类
 * 3. 模板特化：为特定类型提供专门实现 (此处暂不展示)
 *****************************************************************************/

// 函数模板示例：求两个数的较大值
template <typename T>
T myMax(T a, T b)
{
    return (a > b) ? a : b;
}

// 类模板示例：容器类，内部使用 std::vector 来存储
template <typename T>
class Container
{
public:
    void add(T element)
    {
        data_.push_back(element);
    }
    void print() const
    {
        for (auto &el : data_)
        {
            std::cout << el << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<T> data_;
};

/******************************************************************************
 * 第 5 部分：多线程编程
 *
 * 核心概念：
 * 1. 线程：最小的执行单元 (std::thread)
 * 2. 互斥量 (std::mutex)：保护共享资源，防止数据竞争
 * 3. 条件变量 (std::condition_variable)：线程同步机制
 * 4. future/promise：异步任务结果的传递
 * 5. async：高级异步执行，实现简洁的并行
 *****************************************************************************/

// 互斥量：保护共享资源
std::mutex mtx;

// 条件变量：用于实现线程同步
std::condition_variable cv;
bool ready = false;

// 共享资源
int shared_data = 0;

/******************************************************************************
 * 生产者-消费者示例
 * 生产者：写入 shared_data 并通知消费者
 * 消费者：等待通知，读取 shared_data
 *****************************************************************************/

// 生产者函数
void producer()
{
    {
        // std::lock_guard 会在作用域结束时自动解锁
        std::lock_guard<std::mutex> lock(mtx);
        shared_data = 42; // 生产数据
        ready = true;
        std::cout << "[producer] 数据已生产: " << shared_data << std::endl;
    }
    // 通知等待的消费者线程
    cv.notify_one();
}

// 消费者函数
void consumer()
{
    // std::unique_lock 允许显式地锁定和解锁互斥量
    std::unique_lock<std::mutex> lock(mtx);
    // 等待条件满足 (ready == true)
    cv.wait(lock, []
            { return ready; });
    std::cout << "[consumer] 消费到的数据: " << shared_data << std::endl;
}

/******************************************************************************
 * 多线程基础演示
 *****************************************************************************/

// 演示：简单线程函数
void basicThread(int id)
{
    // 演示加锁，保证一次只允许一个线程输出信息
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "Thread " << id << " is running" << std::endl;
}

// 演示：使用 async 执行异步任务
int calculateSum(int a, int b)
{
    // 模拟一个耗时操作
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return a + b;
}

/******************************************************************************
 * main 函数：汇总并演示上述各个模块的特性
 *****************************************************************************/
int main()
{
    std::cout << "=== C++ 进阶特性演示 ===" << std::endl;

    /**************************************************************************
     * 1. 面向对象演示
     **************************************************************************/
    std::cout << "\n--- OOP 演示 ---\n";
    // 演示多态：基类指针指向派生类对象
    Person *person = new Teenager("Alice", 15);
    person->testVirtual(); // 多态调用：实际调用 Teenager::testVirtual()
    person->printInfo();
    // 手动释放对象：会先调用 Teenager 的析构函数，再调用 Person 的析构函数
    delete person;

    /**************************************************************************
     * 2. STL 演示
     **************************************************************************/
    std::cout << "\n--- STL 演示 ---\n";
    demonstrateSTL();

    /**************************************************************************
     * 3. 现代C++特性演示
     **************************************************************************/
    std::cout << "\n--- 现代C++特性演示 ---\n";
    demonstrateModernCpp();

    /**************************************************************************
     * 4. 模板演示
     **************************************************************************/
    std::cout << "\n--- 模板演示 ---\n";
    std::cout << "myMax(10, 20) = " << myMax(10, 20) << std::endl;
    std::cout << "myMax(3.14, 2.72) = " << myMax(3.14, 2.72) << std::endl;

    // 类模板的使用
    Container<int> intContainer;
    intContainer.add(1);
    intContainer.add(2);
    intContainer.add(3);
    std::cout << "intContainer 内容: ";
    intContainer.print();

    /**************************************************************************
     * 5. 多线程演示
     **************************************************************************/
    std::cout << "\n--- 多线程演示 ---\n";

    // (1) 基础线程使用
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i)
    {
        threads.emplace_back(basicThread, i);
    }
    for (auto &t : threads)
    {
        t.join(); // 等待线程执行完毕
    }

    // (2) 生产者-消费者模式
    std::thread prod(producer);
    std::thread cons(consumer);
    prod.join();
    cons.join();

    // (3) async 示例
    std::cout << "\n--- async 示例 ---\n";
    auto future = std::async(std::launch::async, calculateSum, 2, 3);
    std::cout << "主线程：做一些其他工作..." << std::endl;
    // get() 会阻塞主线程，直到异步任务完成并返回结果
    int result = future.get();
    std::cout << "异步计算结果: " << result << std::endl;

    std::cout << "\n=== 程序结束 ===" << std::endl;
    return 0;
}
