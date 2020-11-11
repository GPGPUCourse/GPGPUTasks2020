В этом репозитории предложены задания для [курса по вычислениям на видеокартах в CSC](https://compscicenter.ru/courses/video_cards_computation/2020-autumn/).

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2020/).

# Задание 5. Bitonic sort, radix sort

[![Build Status](https://travis-ci.com/GPGPUCourse/GPGPUTasks2020.svg?branch=task05)](https://travis-ci.com/GPGPUCourse/GPGPUTasks2020)

0. Сделать fork проекта
1. Выполнить задания 5.1 и 5.2
2. Отправить **Pull-request** с названием ```Task05 <Имя> <Фамилия> <Аффиляция>``` (указав вывод каждой программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: начало лекции 26 октября.

Задание 5.1. Bitonic sort
=========

Реализуйте bitonic sort для вещественных чисел (используя локальную память, пока размер подмассива для сортировки не станет слишком большим).

Файлы: ```src/main_bitonic.cpp``` и ```src/cl/bitonic.cl```

Задание 5.2. Radix sort
=========

Реализуйте radix sort для unsigned int (используя локальную память).

Файлы: ```src/main_radix.cpp``` и ```src/cl/radix.cl```
