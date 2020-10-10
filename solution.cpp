#include "defines.h"

// Стандартный ticket lock
typedef struct
{
  // Билет, который будет выдан при следующей попытке захвата блокировки
  unsigned int Ticket;

  // Номер билета читателя, которому разрешено захватить блокировку
  unsigned int CurrentRead;

  // Номер билета писателя, которому разрешено захватить блокировку
  unsigned int CurrentWrite;
} RW_LOCKER;

// Инициализация блокировки
void Setup( RW_LOCKER *Locker )
{
  Locker->Ticket = 0;
  Locker->CurrentRead = 0;
  Locker->CurrentWrite = 0;
}

unsigned int GetTicket( RW_LOCKER *Locker )
{
  // Получение уникального билета
  return atomic_add(&Locker->Ticket, 1);
}

/* Вместо atomic_cmpxchg( ... , (1 << 30), (1 << 30)) можно было использовать atomic_add( ... , 0).
 * (Использование atomic_load условием не разрешено)
 */

int LockReader( RW_LOCKER *Locker, unsigned int CurrentTicket )
{
  return CurrentTicket == atomic_cmpxchg(&Locker->CurrentRead, CurrentTicket, CurrentTicket + 1);
}

void UnlockReader( RW_LOCKER *Locker )
{
  // Увеличиваем CurrentWrite <- Следующий писатель (если это писатель) теперь тоже может войти
  atomic_add(&Locker->CurrentWrite, 1);
}

int LockWriter( RW_LOCKER *Locker, unsigned int CurrentTicket )
{
  return CurrentTicket == atomic_cmpxchg(&Locker->CurrentWrite, (1 << 30), (1 << 30));
}

void UnlockWriter( __local RW_LOCKER *Locker )
{
  // Пропускаем следующий билет
  atomic_add(&Locker->CurrentWrite, 1);
  atomic_add(&Locker->CurrentRead, 1);
}

__kernel do_some_work()
{
    assert(get_group_id == [256, 1, 1]);

    volatile __local RW_LOCKER Locker;

    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
      Setup(&Locker); // Вызываем Setup 1 раз

    barrier(CLK_LOCAL_MEM_FENCE); // Гарантия того, что после этой строки Locker проинициализирован

    __local disjoint_set = ...;

    int LockedFlag;

    for (int iters = 0; iters < 100; ++iters) {      // потоки делают сто итераций
        if (some_random_predicat(get_local_id(0))) { // предикат срабатывает очень редко (например шанс - 0.1%)
          ...                        
            int CurrentTicket = GetTicket(&Locker);
            do
            {
              LockedFlag = LockWriter(&Locker, CurrentTicket);

              if (LockedFlag)
              {
                union(disjoint_set, ...);
                UnlockWriter(&Locker);
              }
              
            } while (!LockedFlag);  // Если LockedFlag то этот поток продолжает крутиться в цикле(ничего не делая), пока остальные не выйдут
          ...
        }
        ...
        do
        {
          LockedFlag = LockReader(&Locker, CurrentTicket);

          if (LockedFlag)
          {
            tmp = get(disjoint_set, ...); // потоки постоянно хотят читать из структурки
            UnlockReader(&Locker);
          }
        } while (!LockedFlag);
        ...
    }
}
