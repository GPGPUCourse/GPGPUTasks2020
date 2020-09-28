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

int GetTicket( RW_LOCKER *Locker )
{
  // ѕолучение уникального билета
  return atomic_add(&Locker->Ticket, 1);
}

/* ¬место atomic_cmpxchg( ... , (1 << 30), (1 << 30)) можно было использовать atomic_add( ... , 0).
 * (»спользование atomic_load условием не разрешено)
 */

void LockReader( RW_LOCKER *Locker )
{
  unsigned int CurrentTicket = GetTicket(Locker);

  // “ак как все потоки в 1 warp читают одновременно, наверное, пойдет и так
  while (CurrentTicket != atomic_cmpxchg(&Locker->CurrentRead, CurrentTicket, CurrentTicket + 1)) // ¬се захват€т блокировку и пойдут читать
  {
  }
}

void UnlockReader( RW_LOCKER *Locker )
{
  // ”величиваем CurrentWrite <- —ледующий писатель (если это писатель) теперь тоже может войти
  atomic_add(&Locker->CurrentWrite, 1);
}

int LockWriter( RW_LOCKER *Locker, unsigned int CurrentTicket )
{
  // Ќикого больше не пускаем
  return CurrentTicket != atomic_cmpxchg(&Locker->CurrentWrite, (1 << 30), (1 << 30));
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
          ...                        
            int CurrentTicket = GetTicket(&Locker);
            int LockedFlag;
            do
            {
              LockedFlag = LockWriter(&Locker, CurrentTicket);

              if (LockedFlag)
              {
                union(disjoint_set, ...);
                UnlockWriter(&Locker);
              }
              
            } while (!LockedFlag);  // ≈сли LockedFlag то этот поток продолжает крутитьс€ в цикле(ничего не дела€), пока остальные не выйдут
          ...
        }
        ...
        LockReader(&Locker);
        tmp = get(disjoint_set, ...); // потоки посто€нно хот€т читать из структурки
        UnlockReader(&Locker);
        ...
    }
}
