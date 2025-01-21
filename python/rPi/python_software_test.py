#Generate the fibonacci sequence up to the nth number
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        print(a)
        a, b = b, a + b

#Display a pyramid of height n using slashes
def pyramid(n):
    base = ' '
    for i in range(n):
        if i == n-1:
            base = '_'
        print(' ' * (n - i - 1) + '/' + base * 2*i + '\\')


if __name__ == '__main__':
    fibonacci(10)
    for i in range(10, 1, -1):
        pyramid(i)