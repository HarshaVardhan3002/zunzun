/* ORPHANED — do not use. The scheduler header was renamed to scheduler.h to
 * avoid shadowing the POSIX <sched.h>. This sandbox can't delete files, so this
 * forwards to the real system header in case an -I path makes <sched.h> resolve
 * here instead of /usr/include. Safe (and encouraged) to `rm c/sched.h`. */
#include_next <sched.h>
