__kernel void do_some_work() {
    assert(
        get_group_id(0) == 256 && 
        get_group_id(1) == 1 && 
        get_group_id(2) == 1
    );

    __local dsu disjoint_set = ...;
    volatile __local int change_ticket_holder = 0;

    for (int iters = 0; iters < 100; ++iters) {                         // thread do some iterations (100 for now)
        const bool need_union = some_random_predicate(get_local_id(0)); // RARELY some of them decide they must change the structure
        int ticket = -1;                                                // ticket for deciding whose turn to change is now

        if (need_union) {                                               // 1. ticket dealing stage
            ticket = atomic_inc(&change_ticket_holder);                 //      1.1. acquire change ticket and increase counter
        }
        barrier(CLK_LOCAL_MEM_FENCE);                                   //      1.2. ensure that no thread goes further ending with temporary value 

        do {                                                            // 2. ordered structure updates
            const int current_turn = change_ticket_holder - 1;          //      2.1. get, whose turn is now
            barrier(CLK_LOCAL_MEM_FENCE);                               //          ensure that no thread is further, so everyone has actual data
            
            if (need_union && ticket + 1 == current_turn) {             //      2.2. if it is current thread's turn, then:
                ...                                                     //          do work
                union(disjoint_set, ...);                               //          (including changes to structure)
                ...

                atomic_dec(&change_ticket_holder);                      //          allow next ticket owner to do its work on next iteration
            }
            barrier(CLK_LOCAL_MEM_FENCE);                               // 3. flush all updates so every thread has same data 
        } while (change_ticket_holder != 0);
        
                                                                        // 4. normal operation with `get`
        tmp = get(disjoint_set, ...);                                   //      threads read something from structure
        ...                                                             //      they do some normal work

        barrier(CLK_LOCAL_MEM_FENCE);                                   // 5. ensure that no thread goes further and MAAYBE changes the structure 
    }
}