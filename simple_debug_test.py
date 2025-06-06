def add_numbers(a, b):
    # 这里可以进入调试
    result = a + b
    print(f'计算: {a} + {b} = {result}')
    return result

def main():
    print('开始测试调试功能')
    # 在下一行设置断点，然后按F11进入函数
    answer = add_numbers(2, 3)
    print(f'最终结果: {answer}')

if __name__ == '__main__':
    main()
