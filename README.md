В этом репозитории предложены задания для [курса по вычислениям на видеокартах в CSC](https://compscicenter.ru/courses/video_cards_computation/2020-autumn/).

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2020/).

# Задание 1. A+B.

[![Build Status](https://travis-ci.com/pavlov200912/GPGPUTasks2020.svg?branch=task01)](https://travis-ci.com/pavlov200912/GPGPUTasks2020)

Задание
=======

0. Сделать fork проекта
1. Прочитать все комментарии подряд и выполнить все **TODO** в файле ``src/main.cpp`` и ``src/cl/aplusb.cl``
2. Отправить **Pull-request** с названием```Task01 <Имя> <Фамилия> <Аффиляция>``` (указав вывод программы при исполнении на вашем компьютере - в тройных кавычках для сохранения форматирования)

**Дедлайн**: начало лекции 21 сентября.

Коментарии
==========

Т.к. в ``TODO 6`` исходники кернела считываются по относительному пути ``src/cl/aplusb.cl``, то нужно правильно настроить working directory. Например в случае CLion нужно открыть ``Edit configurations`` -> и указать ``Working directory: .../НАЗВАНИЕПАПКИПРОЕКТА`` (см. [подробнее](https://github.com/GPGPUCourse/GPGPUTasks2020/tree/task01/.figures))

