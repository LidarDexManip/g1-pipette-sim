"""Device contact reductions; evaluated at every 1 ms physics step."""
import warp as wp


@wp.kernel
def reduce_contacts(nacon:wp.array[int], world:wp.array[int], geom:wp.array[wp.vec2i],
                    dist:wp.array[float], bodies:wp.array[int], category:wp.array[int],
                    pipette:int, force:wp.array[wp.spatial_vector], stats:wp.array2d[float]):
    i = wp.tid()
    if i >= nacon[0]:
        return
    w = world[i]
    wp.atomic_max(stats,w,0,wp.max(0.0,-dist[i]))
    a,b = bodies[geom[i][0]],bodies[geom[i][1]]
    if a == pipette:
        wp.atomic_add(stats,w,category[b],wp.max(0.0,force[i][0]))
    elif b == pipette:
        wp.atomic_add(stats,w,category[a],wp.max(0.0,force[i][0]))


@wp.kernel
def update_peak(stats:wp.array2d[float], peaks:wp.array2d[float]):
    w,c = wp.tid()
    peaks[w,c] = wp.max(peaks[w,c],stats[w,c])


@wp.kernel
def track_overflow(nacon:wp.array[int],nefc:wp.array[int],nconmax:int,njmax:int,flag:wp.array[int]):
    w=wp.tid()
    if nacon[0]>nconmax or nefc[w]>njmax:
        wp.atomic_max(flag,0,1)
