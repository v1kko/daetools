#ifndef DEAL_II_OMP_WORK_STREAM_H
#define DEAL_II_OMP_WORK_STREAM_H

#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dae
{
namespace fe_solver
{
namespace OpenMP_WorkStream
{
template<typename MainClass,
         typename Iterator,
         typename ScratchData,
         typename CopyData>
void run(const Iterator                           &begin,
         const typename identity<Iterator>::type  &end,
         MainClass                                &main_object,
         void (MainClass::*worker)(const Iterator&, ScratchData&, CopyData&),
         void (MainClass::*copier)(const CopyData&),
         const ScratchData                        &scratch_s,
         const CopyData                           &copy_data_s,
         const unsigned int                       queueSize = 32,
         const bool                               verbose = false)
{
#ifdef _OPENMP
        omp_lock_t lock;
        omp_init_lock(&lock);

        // OpenMP does not work with iterators so populate std::vector with all cells.
        // The std::vector supports the random access and can be used with OpenMP.
        std::vector<Iterator> all_iterators;
        //all_iterators.reserve(n_active_cells);
        for(Iterator cell_i = begin; cell_i != end; ++cell_i)
            all_iterators.push_back(cell_i);

        std::vector< boost::shared_ptr<CopyData> > copy_data_queue;
        std::vector< boost::shared_ptr<CopyData> > copy_data_queue_swap;

        copy_data_queue.reserve(queueSize);
        copy_data_queue_swap.reserve(queueSize);

        int n_cells = all_iterators.size();
        #pragma omp parallel
        {
            if(verbose && omp_get_thread_num() == 0)
            {
                printf("Number of threads = %d\n", omp_get_num_threads());
                printf("Queue size        = %d\n", queueSize);
            }

            #pragma omp for schedule(static, 1)
            for(int cellCounter = 0; cellCounter < n_cells; cellCounter++)
            {
                int tid = omp_get_thread_num();

                // Get the cell
                Iterator& cell = all_iterators[cellCounter];
                if(verbose)
                    printf("Thread %d assembling cell %s\n", tid, cell->id().to_string().c_str());

                // Create the scratch and the copy_data objects
                boost::shared_ptr<CopyData> copy_data(new CopyData(copy_data_s));
                ScratchData scratch(scratch_s);

                // Process the cell
                (main_object.*worker)(cell, scratch, *copy_data);

                // Add the copy_data to the queue
                omp_set_lock(&lock);
                    copy_data_queue.push_back(copy_data);
                omp_unset_lock(&lock);

                // When the queue size exceeds the specified queueSize
                // the master thread takes all copy_data objects from the queue
                // and copies the data to the global matrices/array.
                if(tid == 0)
                {
                    if(copy_data_queue.size() >= queueSize)
                    {
                        // Take all objects from the queue and copy them to the global structures.
                        // This way, the other threads do not wait to acquire the omp lock.
                        // The std::vector::swap() function should be fast since it only swaps a couple of pointers.
                        // Anyhow, even copying the shared_ptr objects is cheap.
                        omp_set_lock(&lock);
                            copy_data_queue_swap.swap(copy_data_queue);
                        omp_unset_lock(&lock);

                        // Sort the array so that we always (approximately) copy cell contributions in a sequential order.
                        // This is not always the case, since the master thread often lags behind the others in the team,
                        // (it needs to run the copier function to copy the data).
                        // Therefore, some cells are added later compared to the sequential case.
                        std::sort(copy_data_queue_swap.begin(), copy_data_queue_swap.end());

                        if(verbose)
                            printf("copy_data queue size = %d\n", copy_data_queue_swap.size());
                        for(int k = 0; k < copy_data_queue_swap.size(); k++)
                        {
                            boost::shared_ptr<CopyData>& cd = copy_data_queue_swap[k];
                            (main_object.*copier)(*cd);
                        }

                        // Empty the array after copying
                        copy_data_queue_swap.clear();
                    }
                }
            }
        }

        // If something is left in the queue process it
        if(verbose)
            printf("Number of copy_data objects left in the queue after assembling = %d\n", copy_data_queue.size());
        for(int k = 0; k < copy_data_queue.size(); k++)
        {
            boost::shared_ptr<CopyData>& cd = copy_data_queue[k];
            (main_object.*copier)(*cd);
        }
#else
    // If -fopenmp compiler flag is not specified and _OPENMP not defined - do everything sequentially

    if(verbose)
        printf("OpenMP is disabled - working sequentally\n");

    CopyData    copy_data(copy_data_s);
    ScratchData scratch  (scratch_s);

    for(Iterator cell = begin; cell != end; ++cell)
    {
        // Process the cell
        (main_object.*worker)(cell, scratch, copy_data);

        // Copies the data to the global matrices/array.
        (main_object.*copier)(copy_data);
    }
#endif
}

}
}
}

#endif
