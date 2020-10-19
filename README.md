В этом репозитории предложены задания для [курса по вычислениям на видеокартах в CSC](https://compscicenter.ru/courses/video_cards_computation/2020-autumn/).

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2020/).

# Задание 42. Система непересекающихся множеств и барьеры (необязательное)

0. Сделать fork проекта
1. Выполнить задание ниже
2. Отправить **Pull-request** с названием```Task42 <Имя> <Фамилия> <Аффиляция>``` и укажите текстом PR дополненный вашим решением набросок кода предложенный ниже - обрамив тройными кавычками с указанием ```C++``` языка [после первой тройки кавычек](https://docs.github.com/en/free-pro-team@latest/github/writing-on-github/creating-and-highlighting-code-blocks#fenced-code-blocks) (см. [пример](https://github.com/GPGPUCourse/GPGPUTasks2020/blame/b544d77cd4bc96b92b4a62d1eaaebf05075bf582/README.md#L45-L63)).

**Дедлайн**: начало лекции 12 октября. Но задание необязательное и за него можно получить всего лишь один бонусный балл.

Локальная структура у рабочей группы
=========

У каждой рабочей группы своя [СНМ (система непересекающихся множеств)](https://neerc.ifmo.ru/wiki/index.php?title=%D0%A1%D0%9D%D0%9C_(%D1%80%D0%B5%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_%D1%81_%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E_%D0%BB%D0%B5%D1%81%D0%B0_%D0%BA%D0%BE%D1%80%D0%BD%D0%B5%D0%B2%D1%8B%D1%85_%D0%B4%D0%B5%D1%80%D0%B5%D0%B2%D1%8C%D0%B5%D0%B2)).

Важно лишь понимать что у **СНМ** есть две операции:

 - ```union()``` - НЕ thread-safe т.к. может перелопатить вообще всю структурку (переподвесить все элементы дерева)

 - ```get()``` - функция только читает данные из disjoint_set - thread-safe лишь если параллельно не выполняется ```union()```

В целом это может быть любая другая структура с функцией чтения и модификации затрагивающей потенциально всю структуру.

Поведение потоков
=========

Потоки часто читают и очень редко пишут.

Нельзя допустить чтобы когда кто-то пишет (вызывает ```union()```) - кто-то читал (вызывал ```get()```).

Нельзя допустить чтобы когда кто-то пишет (вызывает ```union()```) - кто-то еще писал (вызывал ```union()```).

У вас есть
=========

Можно пользоваться барьерами на локальную группу и операциями [atomic_add](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/atomic_add.html) и [atomic_cmpxchg](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/atomic_cmpxchg.html) над локальными переменными.

Задание
=========

Дополните набросок кода ниже так, чтобы не было гонок (желательно добавить поясняющие комментарии почему вам кажется что это работает):

```C++
#define WORP_SIZE ...
#define WORPS_AMOUNT ...
// get_local_size(0) == WORP_SIZE * WORPS_AMOUNT

__kernel do_some_work()
{
    assert(get_group_id == [256, 1, 1]);

    __local disjoint_set = ...;

    __local int true_pred_sum[WORPS_AMOUNT];
    __local int modifying = 0;

    int worp_id = get_local_id(0) % WORP_SIZE;
    int worp_num = get_local_id(0) / WORP_SIZE;

    for (int iters = 0; iters < 100; ++iters) {
        if (worp_id == 0) {
            true_pred_sum[worp_num] = 0;
        }
        // проинициализировали массив true_pred_sum

        int true_pred = 0;
        for (int id = worp_id; id < get_local_size(0); id += WORP_SIZE) {
            true_pred += some_random_predicat(id);
        }
        // узнали для каких-то потоков каждого ворпа, есть ли какие-то потоки в других ворпах с истинным предикатом

        if (true_pred) {
            atomic_add(&true_pred_sum[worp_num], true_pred);
        }
        // теперь каждый элемент true_pred_sum - суммарное количество необходимых union-ов

        if (true_pred_sum[worp_num] && iters > 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
            // подождали, чтобы избежать WAR-гонки
        }

        int pred = some_random_predicat(get_local_id(0));
        // pred говорит, нужно ли ещё этому потоку делать union
        while (true_pred_sum[0] > 0) {
            // крутимся, пока все потоки workgroup-ы не внесут свои изменения
            while (pred) {
                // зашедший сюда поток ждёт своей очереди, чтобы сделать union
                if (atomic_cmpxchg(&modifying, 0, 1) == 0) {
                    // сюда в workgroup-е одновременно зайдёт только один поток
                    ...
                    union(disjoint_set, ...);
                    ...
                    true_pred_sum[0]--;
                    modifying = 0;
                    pred = false;
                }
            }
        }
        // после выхода из while-а все изменения внесены; можно безопасно читать

        ...
        tmp = get(disjoint_set, ...);
        ...
    }
}
```

**Подсказка**: если вы придумали решение, попробуйте подумать на тему "раз указатель на инструкцию у ворпа один и тот же, не приведет ли это к проблемам в барьерах/атомарных операциях?"
