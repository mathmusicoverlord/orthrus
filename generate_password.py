if __name__ == '__main__':

    # generate password
    import string
    import random
    from itertools import permutations
    import numpy as np

    def factorial(n):
        # single line to find factorial
        return 1 if (n == 1 or n == 0) else n * factorial(n - 1);
        return 1 if (n == 1 or n == 0) else n * factorial(n - 1);

    password = []
    #
    for i in range(13):
        password.append(random.choice(string.ascii_letters))

    for i in range(4):
        password.append(random.choice(string.digits))

    for i in range(3):
        password.append(random.choice(string.punctuation))

    password = ''.join(password)

    # permute a random number of times
    i0 = np.random.randint(0, 10_000)
    for i, new_password in enumerate(permutations(password)):
        password_temp = new_password
        if i == i0:
            break

    password = password_temp
    password = ''.join(password)
    print('Password: ' + password)


