#include "defines.h"

// —тандартный ticket lock
typedef struct
{
  // Ѕилет, который будет выдан при следующей попытке захвата блокировки
  unsigned int Ticket;

  // Ќомер билета читател€, которому разрешено захватить блокировку
  unsigned int CurrentRead;

  // Ќомер билета писател€, которому разрешено захватить блокировку
  unsigned int CurrentWrite;
} RW_LOCKER;

// »нициализаци€ блокировки
void Setup( RW_LOCKER *Locker )
{
  Locker->Ticket = 0;
  Locker->CurrentRead = 0;
  Locker->CurrentWrite = 0;
}

/* ¬место atomic_cmpxchg( ... , (1 << 30), (1 << 30)) можно было использовать atomic_add( ... , 0).
 * (»спользование atomic_load условием не разрешено)
 */

void LockReader( RW_LOCKER *Locker )
{
  // ѕолучение уникального билета
  unsigned int CurrentTicket = atomic_add(&Locker->Ticket, 1);

  // ∆дем пока нам разрешат войти
  while (CurrentTicket != atomic_cmpxchg(&Locker->CurrentRead, (1 << 30), (1 << 30))) // CurrentRead никогда не будет равен (1 << 30) (ј даже если будет, то на его место снова запишетс€ (1 << 30)) <- Ёто эквивалентно атомарной загрузке CurrentRead
  {
  }

  // ”величиваем CurrentRead <- —ледующий читатель (если это читатель) может войти
  atomic_add(&Locker->CurrentRead, 1);
}

void UnlockReader( RW_LOCKER *Locker )
{
  // ”величиваем CurrentWrite <- —ледующий писатель (если это писатель) теперь тоже может войти
  atomic_add(&Locker->CurrentWrite, 1);
}

void LockWriter( RW_LOCKER *Locker )
{
  // ѕолучение уникального билета
  unsigned int CurrentTicket = atomic_add(&Locker->Ticket, 1);

  // ∆дем пока нам разрешат войти
  while (CurrentTicket != atomic_cmpxchg(&Locker->CurrentWrite, (1 << 30), (1 << 30))) // CurrentWrite никогда не будет равен (1 << 30) (ј даже если будет, то на его место снова запишетс€ (1 << 30)) <- Ёто эквивалентно атомарной загрузке CurrentWrite
  {
  }

  // Ќикого больше не пускаем
}

void UnlockWriter( __local RW_LOCKER *Locker )
{
  // ѕропускаем следующий билет
  atomic_add(&Locker->CurrentWrite, 1);
  atomic_add(&Locker->CurrentRead, 1);
}

__kernel do_some_work()
{
    assert(get_group_id == [256, 1, 1]);

    volatile __local RW_LOCKER Locker;

    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
      Setup(&Locker); // ¬ызываем Setup 1 раз

    barrier(CLK_LOCAL_MEM_FENCE); // √аранти€ того, что после этой строки Locker проинициализирован

    __local disjoint_set = ...;

    for (int iters = 0; iters < 100; ++iters) {      // потоки делают сто итераций
        if (some_random_predicat(get_local_id(0))) { // предикат срабатывает очень редко (например шанс - 0.1%)
            ...                        // на каждой итерации некоторые потоки
            LockWriter(&Locker);
            union(disjoint_set, ...);  // могут захотеть обновить нашу структурку
            UnlockWriter(&Locker);
            ...
        }
        ...
        LockReader(&Locker);
        tmp = get(disjoint_set, ...); // потоки посто€нно хот€т читать из структурки
        UnlockReader(&Locker);
        ...
    }
}
