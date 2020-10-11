В этом репозитории предложены задания для [курса по вычислениям на видеокартах в CSC](https://compscicenter.ru/courses/video_cards_computation/2020-autumn/).

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2020/).

# Задание 4. Транспонирование матрицы, умножение матриц.

[![Build Status](https://travis-ci.com/GPGPUCourse/GPGPUTasks2020.svg?branch=task04)](https://travis-ci.com/GPGPUCourse/GPGPUTasks2020)

0. Сделать fork проекта
1. Выполнить задания 4.1 и 4.2 ниже
2. Отправить **Pull-request** с названием```Task04 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: начало лекции 12 октября.

Если времени не хватит - отправьте то что вы успели сделать с комментарием что вы хотите дополнительную неделю на это задание
(и мне очень поможет если вы сможете детализировать на что у вас ушло слишком много времени, сколько в целом вы времени потратили и т.п.).

Задание 4.1. Транспонирование матрицы
=========

Реализуйте транспонирование матрицы таким образом, чтобы доступ и на чтение и на запись к глобальной видеопамяти был coalesced. (т.е. через локальную память)

Файлы: ```src/main_matrix_transpose.cpp``` и ```src/cl/matrix_transpose.cl```

Задание 4.2. Умножение матриц
=========

Реализуйте умножение матриц через локальную память. (на лекции это вплоть до "Умножение матриц 2: локальная память")

Файлы: ```src/main_matrix_multiplication.cpp``` и ```src/cl/matrix_multiplication.cl```
