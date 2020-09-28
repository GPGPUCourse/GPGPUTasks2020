#include "defines.h"

// Ñòàíäàðòíûé ticket lock
typedef struct
{
  // Áèëåò, êîòîðûé áóäåò âûäàí ïðè ñëåäóþùåé ïîïûòêå çàõâàòà áëîêèðîâêè
  unsigned int Ticket;

  // Íîìåð áèëåòà ÷èòàòåëÿ, êîòîðîìó ðàçðåøåíî çàõâàòèòü áëîêèðîâêó
  unsigned int CurrentRead;

  // Íîìåð áèëåòà ïèñàòåëÿ, êîòîðîìó ðàçðåøåíî çàõâàòèòü áëîêèðîâêó
  unsigned int CurrentWrite;
} RW_LOCKER;

// Èíèöèàëèçàöèÿ áëîêèðîâêè
void Setup( RW_LOCKER *Locker )
{
  Locker->Ticket = 0;
  Locker->CurrentRead = 0;
  Locker->CurrentWrite = 0;
}

int GetTicket( RW_LOCKER *Locker )
{
  // Ïîëó÷åíèå óíèêàëüíîãî áèëåòà
  return atomic_add(&Locker->Ticket, 1);
}

/* Âìåñòî atomic_cmpxchg( ... , (1 << 30), (1 << 30)) ìîæíî áûëî èñïîëüçîâàòü atomic_add( ... , 0).
 * (Èñïîëüçîâàíèå atomic_load óñëîâèåì íå ðàçðåøåíî)
 */

void LockReader( RW_LOCKER *Locker )
{
  unsigned int CurrentTicket = GetTicket(Locker);

  // Òàê êàê âñå ïîòîêè â 1 warp ÷èòàþò îäíîâðåìåííî, íàâåðíîå, ïîéäåò è òàê
  while (CurrentTicket != atomic_cmpxchg(&Locker->CurrentRead, CurrentTicket, CurrentTicket + 1)) // Âñå çàõâàòÿò áëîêèðîâêó è ïîéäóò ÷èòàòü
  {
  }
}

void UnlockReader( RW_LOCKER *Locker )
{
  // Óâåëè÷èâàåì CurrentWrite <- Ñëåäóþùèé ïèñàòåëü (åñëè ýòî ïèñàòåëü) òåïåðü òîæå ìîæåò âîéòè
  atomic_add(&Locker->CurrentWrite, 1);
}

int LockWriter( RW_LOCKER *Locker, unsigned int CurrentTicket )
{
  // Íèêîãî áîëüøå íå ïóñêàåì
  return CurrentTicket == atomic_cmpxchg(&Locker->CurrentWrite, (1 << 30), (1 << 30));
}

void UnlockWriter( __local RW_LOCKER *Locker )
{
  // Ïðîïóñêàåì ñëåäóþùèé áèëåò
  atomic_add(&Locker->CurrentWrite, 1);
  atomic_add(&Locker->CurrentRead, 1);
}

__kernel do_some_work()
{
    assert(get_group_id == [256, 1, 1]);

    volatile __local RW_LOCKER Locker;

    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0)
      Setup(&Locker); // Âûçûâàåì Setup 1 ðàç

    barrier(CLK_LOCAL_MEM_FENCE); // Ãàðàíòèÿ òîãî, ÷òî ïîñëå ýòîé ñòðîêè Locker ïðîèíèöèàëèçèðîâàí

    __local disjoint_set = ...;

    for (int iters = 0; iters < 100; ++iters) {      // ïîòîêè äåëàþò ñòî èòåðàöèé
        if (some_random_predicat(get_local_id(0))) { // ïðåäèêàò ñðàáàòûâàåò î÷åíü ðåäêî (íàïðèìåð øàíñ - 0.1%)
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
              
            } while (!LockedFlag);  // Åñëè LockedFlag òî ýòîò ïîòîê ïðîäîëæàåò êðóòèòüñÿ â öèêëå(íè÷åãî íå äåëàÿ), ïîêà îñòàëüíûå íå âûéäóò
          ...
        }
        ...
        LockReader(&Locker);
        tmp = get(disjoint_set, ...); // ïîòîêè ïîñòîÿííî õîòÿò ÷èòàòü èç ñòðóêòóðêè
        UnlockReader(&Locker);
        ...
    }
}
