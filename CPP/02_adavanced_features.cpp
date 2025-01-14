/******************************************************************************
 * 文件名：02_advanced_features.cpp
 *
 * 该示例演示了 C++ 中的以下进阶特性：
 * 1. 异常处理机制
 * 2. 面向对象编程 (OOP)
 * 3. STL 容器和算法（含 stack 与 queue）
 * 4. Lambda 表达式和现代 C++ 特性
 * 5. 模板和泛型编程
 * 6. 多线程编程基础
 * 7. 运算符重载
 * 8. 文件操作
 *
 * 在学习该示例的过程中，建议先了解 C++ 基础语法，如基本数据类型、
 * 指针与引用、控制流程、函数等，再来逐步深入。
 *****************************************************************************/

#include <iostream>             // 标准输入输出流
#include <string>               // 字符串
#include <vector>               // 动态数组容器
#include <map>                  // 键值对容器
#include <set>                  // 集合容器
#include <list>                 // 双向链表容器
#include <stack>                // 栈容器适配器
#include <queue>                // 队列容器适配器
#include <algorithm>            // 常用算法，如 sort、find
#include <stdexcept>            // 标准异常类
#include <thread>               // 线程库
#include <mutex>                // 互斥量
#include <condition_variable>   // 条件变量
#include <future>               // std::async、std::future
#include <chrono>               // 时间库
#include <fstream>              // 文件流
#include <sstream>              // 字符串流

/******************************************************************************
 * 第 1 部分：异常处理
 * 
 * C++ 中的异常处理通过 try-catch 机制实现，可以捕获并处理在运行时
 * 抛出的异常对象。throw 可以抛出异常对象，catch 通过类型匹配捕获异常。
 *****************************************************************************/

// 自定义文件操作异常，继承自 std::runtime_error
class FileError : public std::runtime_error {
public:
    FileError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

// 自定义年龄验证异常，继承自 std::runtime_error
class InvalidAgeError : public std::runtime_error {
public:
    InvalidAgeError(const std::string& msg) 
        : std::runtime_error(msg) {}
};

/******************************************************************************
 * 第 2 部分：面向对象编程 (OOP)
 * 
 * C++ 中的继承、封装、多态是 OOP 的核心。下面代码中：
 * 1. Person 是一个抽象基类（含纯虚函数 work()）。
 * 2. Teenager 和 Student 分别继承自 Person。
 * 3. 演示了虚析构函数、虚函数覆盖、运算符重载、友元函数等。
 *****************************************************************************/

// 定义基类 Person（抽象类）
class Person {
public:
    // 构造函数，加入年龄验证
    Person(const std::string& name, int age) 
        : name_(name), age_(age) 
    {
        // 如果年龄不合理，则抛出自定义异常
        if (age < 0 || age > 150) {
            throw InvalidAgeError("Age must be between 0 and 150");
        }
        std::cout << "[Person 构造函数] name = " << name_
                  << ", age = " << age_ << std::endl;
    }

    // 虚析构函数：保证派生类对象通过基类指针删除时能正确调用派生类的析构函数
    virtual ~Person() {
        std::cout << "[Person 析构函数]" << std::endl;
    }

    // 纯虚函数，使 Person 成为抽象类（不能直接实例化）
    virtual void work() const = 0;

    // 虚函数：打印对象信息，可在派生类中覆盖
    virtual void printInfo() const {
        std::cout << "Name: " << name_
                  << ", Age: " << age_ << std::endl;
    }

    // 运算符重载：演示 < 与 == 的重载
    bool operator<(const Person& other) const {
        return age_ < other.age_;
    }

    bool operator==(const Person& other) const {
        return (age_ == other.age_) && (name_ == other.name_);
    }

    // 友元函数：输出流运算符重载，通过友元可以访问 private 成员
    friend std::ostream& operator<<(std::ostream& os, const Person& person);

protected:
    std::string name_;  // 姓名
    int age_;           // 年龄
};

// 友元函数：输出运算符重载
std::ostream& operator<<(std::ostream& os, const Person& person) {
    os << "Person[name=" << person.name_ 
       << ", age=" << person.age_ << "]";
    return os;
}

// 派生类 Teenager，继承自 Person
class Teenager : public Person {
public:
    Teenager(const std::string& name, int age)
        : Person(name, age) {
        std::cout << "[Teenager 构造函数]" << std::endl;
    }

    ~Teenager() override {
        std::cout << "[Teenager 析构函数]" << std::endl;
    }

    // 必须实现纯虚函数 work()
    void work() const override {
        std::cout << name_ << " is studying in high school" << std::endl;
    }
};

// 派生类 Student，继承自 Person
class Student : public Person {
public:
    Student(const std::string& name, int age, const std::string& major)
        : Person(name, age), major_(major) {}

    // 实现纯虚函数 work()
    void work() const override {
        std::cout << name_ << " is studying " << major_ << std::endl;
    }

    // 覆盖基类的 printInfo()
    void printInfo() const override {
        Person::printInfo();
        std::cout << "Major: " << major_ << std::endl;
    }

private:
    std::string major_;  // 专业
};

/******************************************************************************
 * 第 3 部分：文件操作
 * 
 * 通过 <fstream> 提供的 ifstream、ofstream、fstream 等类进行文件读写。
 * 下例中 FileManager 封装了文件的读写操作，并在失败时抛出自定义异常 FileError。
 *****************************************************************************/

class FileManager {
public:
    // 将 Person* 向量中的内容写入文件
    static void writeToFile(const std::string& filename, const std::vector<Person*>& people) {
        std::ofstream file(filename);
        if (!file) {
            throw FileError("Cannot open file for writing: " + filename);
        }
        
        // 将每个 Person* 的内容写到文件
        for (const auto& person : people) {
            file << *person << "\n";
        }
    }
    
    // 从文件中读取所有内容并返回 string
    static std::string readFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw FileError("Cannot open file for reading: " + filename);
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
};

/******************************************************************************
 * 第 4 部分：STL (Standard Template Library)
 * 
 * 1. 容器（如 vector、map、set、list、stack、queue）
 * 2. 常用算法（如 sort、find、erase 等）
 * 3. 迭代器与范围 for 循环
 *****************************************************************************/

void demonstrateSTL() {
    std::cout << "[STL 容器与算法演示 - CRUD 示例]" << std::endl;

    // 1. vector：动态数组
    std::vector<int> vec = {1, 2, 3};
    std::cout << "vector 初始化内容: ";
    for (auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // vector 的增删改操作
    vec.push_back(4);  // 末尾添加元素
    vec[0] = 10;       // 修改第一个元素
    vec.pop_back();    // 弹出末尾元素

    std::cout << "vector 修改后内容: ";
    for (auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;

    // 2. map：键值对容器
    std::map<std::string, int> ages;
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    
    std::cout << "map 内容: ";
    for (auto& kv : ages) {
        std::cout << kv.first << ":" << kv.second << " ";
    }
    std::cout << std::endl;
    
    // 修改和删除
    ages["Alice"] = 26;  // 修改
    ages.erase("Bob");   // 删除

    std::cout << "map 修改后内容: ";
    for (auto& kv : ages) {
        std::cout << kv.first << ":" << kv.second << " ";
    }
    std::cout << std::endl << std::endl;

    // 3. set：唯一元素集合
    std::set<int> numSet = {3, 1, 4, 1, 5};
    std::cout << "set 内容: ";
    for (auto& val : numSet) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 查找与插入删除
    if (numSet.find(3) != numSet.end()) {
        numSet.erase(3);   // 删除
        numSet.insert(33); // 插入
    }
    numSet.erase(1);

    std::cout << "set 修改后内容: ";
    for (auto& val : numSet) {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;

    // 4. list：双向链表容器
    std::list<int> myList = {10, 20, 30};
    std::cout << "list 内容: ";
    for (auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // list 的增删改
    myList.push_front(5);         // 头部插入
    *(++myList.begin()) = 15;     // 修改第二个元素
    myList.pop_back();            // 删除末尾元素

    std::cout << "list 修改后内容: ";
    for (auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;

    // 5. stack：栈（后进先出 - LIFO）
    std::stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    std::cout << "stack 顶部元素: " << s.top() << std::endl;
    s.pop();
    std::cout << "stack 弹出一个元素后，新的顶部元素: " << s.top() << std::endl << std::endl;

    // 6. queue：队列（先进先出 - FIFO）
    std::queue<int> q;
    q.push(10);
    q.push(20);
    q.push(30);
    std::cout << "queue 队首元素: " << q.front() << ", 队尾元素: " << q.back() << std::endl;
    q.pop();
    std::cout << "queue 弹出一个元素后，新队首元素: " << q.front() << std::endl << std::endl;

    // 常用算法演示
    std::vector<int> algoVec = {4, 1, 3, 2, 5};
    std::sort(algoVec.begin(), algoVec.end());  // 排序
    auto it = std::find(algoVec.begin(), algoVec.end(), 3);  // 查找
    if (it != algoVec.end()) {
        std::cout << "在 algoVec 中找到了 3" << std::endl;
    }

    std::cout << "algoVec 排序后: ";
    for (auto& val : algoVec) {
        std::cout << val << " ";
    }
    std::cout << std::endl << std::endl;
}

/******************************************************************************
 * 第 5 部分：Lambda 表达式和现代 C++ 特性
 * 
 * 1. lambda 表达式：可在函数内部临时定义函数
 * 2. auto 关键字：自动类型推导
 * 3. 范围 for 循环
 *****************************************************************************/

void demonstrateModernCpp() {
    std::cout << "[Lambda 表达式与现代 C++ 特性]" << std::endl;

    // 1. 简单 Lambda 示例
    auto add = [](int a, int b) { return a + b; };
    int sum = add(3, 5);
    std::cout << "add(3, 5) = " << sum << std::endl;

    // 2. 带捕获的 Lambda
    int multiplier = 10;
    auto multiply = [multiplier](int x) { return x * multiplier; };
    std::cout << "multiply(4) = " << multiply(4) << std::endl;

    // 3. 范围 for 循环
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "numbers: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl << std::endl;
}

/******************************************************************************
 * 第 6 部分：模板与泛型编程
 * 
 * 函数模板与类模板是 C++ 泛型编程的核心，可使代码具有更高的复用性。
 *****************************************************************************/

// 函数模板示例：返回两者中较大值
template <typename T>
T myMax(T a, T b) {
    return (a > b) ? a : b;
}

// 类模板示例：简单的安全数组包装
template<typename T>
class SafeArray {
public:
    // 构造函数动态申请数组
    SafeArray(size_t size) 
        : size_(size), data_(new T[size]) {}

    // 析构函数释放资源
    ~SafeArray() { 
        delete[] data_; 
    }
    
    // 重载下标运算符 (非 const 版本)
    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[index];
    }
    
    // 重载下标运算符 (const 版本)
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[index];
    }

private:
    size_t size_;
    T* data_;
    
    // 为简化示例，禁止拷贝和赋值
    SafeArray(const SafeArray&) = delete;
    SafeArray& operator=(const SafeArray&) = delete;
};

// 容器类模板：使用 std::vector 来存储任意类型数据
template <typename T>
class Container {
public:
    void add(T element) {
        data_.push_back(element);
    }

    void print() const {
        for (auto& el : data_) {
            std::cout << el << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<T> data_;
};

/******************************************************************************
 * 第 7 部分：多线程编程
 * 
 * C++11 提供了 <thread>、<mutex>、<condition_variable> 等库，可实现多线程
 * 编程。以下示例展示了生产者-消费者模型、线程同步与通信、以及 std::async 等。
 *****************************************************************************/

std::mutex mtx;
std::condition_variable cv;
bool ready = false;  // 表示数据是否就绪
int shared_data = 0; // 共享数据

// 生产者线程函数
void producer() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shared_data = 42;
        ready = true;
        std::cout << "[producer] 数据已生产: " << shared_data << std::endl;
    }
    // 唤醒等待的消费者线程
    cv.notify_one();
}

// 消费者线程函数
void consumer() {
    std::unique_lock<std::mutex> lock(mtx);
    // 等待直到 ready 为 true
    cv.wait(lock, [] { return ready; });
    std::cout << "[consumer] 消费到的数据: " << shared_data << std::endl;
}

// 简单线程函数示例
void basicThread(int id) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "Thread " << id << " is running" << std::endl;
}

// 演示 std::async 和 std::future
int calculateSum(int a, int b) {
    // 模拟耗时操作
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return a + b;
}

/******************************************************************************
 * main 函数：汇总并演示上述各个模块的特性
 *****************************************************************************/

int main() {
    std::cout << "=== C++ 进阶特性演示 ===" << std::endl;

    // ---------------------
    // 1. 异常处理 + OOP 演示
    // ---------------------
    std::cout << "\n--- 异常处理和 OOP 演示 ---\n";
    try {
        // 尝试创建一个年龄非法的对象
        Person* invalidPerson = new Student("Invalid", -1, "CS");
        // 如果上面未抛出异常，则需要 delete
        delete invalidPerson;
    } catch (const InvalidAgeError& e) {
        std::cerr << "Age error: " << e.what() << std::endl;
    }

    // 创建一些 Person 派生类对象
    std::vector<Person*> people;
    people.push_back(new Student("Alice", 20, "Computer Science"));
    people.push_back(new Teenager("Bob", 15));

    // 演示虚函数调用
    for (const auto& person : people) {
        person->work();       // 多态调用
        person->printInfo();  // 多态调用
        std::cout << *person << std::endl; // 调用友元函数重载的输出运算符
    }

    // ---------------------
    // 2. 文件操作演示
    // ---------------------
    std::cout << "\n--- 文件操作演示 ---\n";
    try {
        FileManager::writeToFile("people.txt", people);
        std::cout << "File content:\n"
                  << FileManager::readFromFile("people.txt");
    } catch (const FileError& e) {
        std::cerr << "File error: " << e.what() << std::endl;
    }

    // ---------------------
    // 3. STL 演示
    // ---------------------
    std::cout << "\n--- STL 演示 ---\n";
    demonstrateSTL();

    // ---------------------
    // 4. 现代C++特性演示
    // ---------------------
    std::cout << "\n--- 现代C++特性演示 ---\n";
    demonstrateModernCpp();

    // ---------------------
    // 5. 模板演示
    // ---------------------
    std::cout << "\n--- 模板演示 ---\n";
    std::cout << "myMax(10, 20) = " << myMax(10, 20) << std::endl;
    std::cout << "myMax(3.14, 2.72) = " << myMax(3.14, 2.72) << std::endl;

    // 演示容器类模板
    Container<int> intContainer;
    intContainer.add(1);
    intContainer.add(2);
    intContainer.add(3);
    std::cout << "intContainer 内容: ";
    intContainer.print();

    // 演示 SafeArray
    SafeArray<int> numbers(5);
    for (size_t i = 0; i < 5; ++i) {
        numbers[i] = static_cast<int>(i * 10);
    }
    std::cout << "SafeArray 内容: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;

    // 故意触发越界异常
    try {
        std::cout << numbers[10] << std::endl; // 应该抛出 std::out_of_range
    } catch (const std::out_of_range& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    // ---------------------
    // 6. 多线程编程演示
    // ---------------------
    std::cout << "\n--- 多线程编程演示 ---\n";

    // 6.1 基础线程示例
    std::thread t1(basicThread, 1);
    std::thread t2(basicThread, 2);

    // 6.2 生产者-消费者模型
    std::thread prod(producer);
    std::thread cons(consumer);

    // 6.3 std::async 和 std::future
    std::future<int> result = std::async(std::launch::async, calculateSum, 10, 20);
    std::cout << "正在计算 10 + 20..." << std::endl;
    std::cout << "10 + 20 = " << result.get() << std::endl; // get() 会等待线程完成并获取返回值

    // 等待所有线程结束
    t1.join();
    t2.join();
    prod.join();
    cons.join();

    // 释放 people 中的动态对象
    for (auto& person : people) {
        delete person;
    }

    std::cout << "\n=== 程序结束 ===" << std::endl;
    return 0;
}
