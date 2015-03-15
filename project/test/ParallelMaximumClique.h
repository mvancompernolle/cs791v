/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#ifndef HEADER_DEFAAFC421A53A4F
#define HEADER_DEFAAFC421A53A4F

#include "MaximumCliqueBase.h"
#include <thread>
#include <pthread.h>
#include <memory>
#include <deque> // TODO: test deque vs. vector for storing a queue of jobs


using std::swap;
    
    
#define TRACE_MASK_THREAD 0x10
#define TRACE_MASK_CLIQUE 0x20


/**
    VertexRepresentation should be
        of the most efficient scalar type on the given hardware
    VertexSetRepresentation must support:
        resize(size)
        reserve(max_size)
        add(value) // vector.push_back or set.operator[]
        pop(value) // takes last added element (that equals to v) from the set
        pop() // returns the value of the last added element, and also removes it from the set (only in vectors)
        uint size()
        operator[] (index)
        clear() // erase all elements
        TODO: list others
    Vector should be a template class (like std::vector) for dynamicly-sized arrays 
    
    function intersect(VertexSet, VertexId, VertexSet& result)
**/
template<
    class Vertex_t,
    class Set_t, 
    class Graph_t, 
    class Sort_t,
    template<class S> class InitialSort_t
>
class ParallelMaximumCliqueProblem : public InitialSort_t<Sort_t> {
public:
    typedef Vertex_t VertexId;
    typedef Set_t VertexSet;
    typedef Graph_t Graph;
    typedef Sort_t Sorter;
    typedef InitialSort_t<Sort_t> InitialSorter;
    typedef typename Sorter::NumberedSet NumberedSet;
    
    struct Job {
        VertexSet c;            // initial clique
        VertexSet vertices;     // available vertices
        NumberedSet numbers;    // available vertices
        unsigned int estimatedMax;       // used to queue jobs from the most perspective down to the least perspective
        
        Job()  {}
        Job(const VertexSet& cin, const VertexSet& vin, const NumberedSet& nin, int estimate) : c(cin), vertices(vin), numbers(nin), estimatedMax(estimate)  {}
        void set(const VertexSet& cin, const VertexSet& vin, const NumberedSet& nin, int estimate) {c = cin; vertices = vin; numbers = nin; estimatedMax = estimate;}
        friend void swap(Job& a, Job& b) {
            std::swap(a.c, b.c); 
            std::swap(a.vertices, b.vertices);
            std::swap(a.numbers, b.numbers);
            std::swap(a.estimatedMax, b.estimatedMax);
        }
    };

    struct Worker : Sorter {
        // parent pMCP - the problem for which this worker is working
        ParallelMaximumCliqueProblem* parent;
        // Timer and counters
        PrecisionTimer timer;
        unsigned long long int steps;
        // the same graph as the parent pMCP
        Graph* graph;
        // current clique
        VertexSet c;
        // id of the worker (helps with the end-time statistics)
        int id;
        // copy of parent maxSize (size of the active maximumClique)
        unsigned int localMaxSize;
        // break the work on a job when going below the specified level (size of a clique); search will not continue on a level lower than jobLevelBreak
        unsigned int jobLevelBreak;
        
        using Sorter::notEmpty;
        using Sorter::topNumber;
        using Sorter::topVertex;
        using Sorter::popTop;
        using Sorter::numberSort;
        
        Worker() : graph(nullptr) {}
        
        void setup(ParallelMaximumCliqueProblem* pmcp, int newId) {
            parent = pmcp;
            graph = pmcp->graph;
            id = newId;
            Sorter::init(graph);
            steps = 0;
        }
        
        void threadFunc() {
            localMaxSize = parent->maxSize;
            TRACE("threadFunc start", TRACE_MASK_THREAD, 1);
            int jobsDone = 0;
            try {
                { // this scope is for the scope timer in the next line: the scope must end before the function sends the results to the parent thread
                    ScopeTimer t(timer);
                    // work while there are jobs available
                    TRACE("threadFunc: getting a job", TRACE_MASK_THREAD, 1);
                    while (true) {
                        Job job;
                        // grab a job from the queue
                        {  
                            std::lock_guard<std::mutex> lk(parent->mutexJobs); 
                            TRACEVAR(parent->activeJobs, TRACE_MASK_THREAD, 2);
                            // if there is no queued jobs then wait; if there is also no active jobs (that would be adding to the queue) then quit
                            if (parent->jobs.size() == 0) {
                                if (parent->activeJobs == 0)
                                    break;
                                else
                                    continue;
                            }
                            
                            // take a job from the top of the queue, check if a new job can be added to the queue immediately 
                            if ((parent->jobs.size() > parent->maxJobs) || (parent->jobs.back().numbers.size() < 5)) {
                                // job on the queue is a small one; do not split it further (the number to check against is arbitrary)
                                // (swap is the fastest operation, job is empty now)
                                swap(job, parent->jobs.back());
                                parent->jobs.pop_back();
                                parent->requireMoreJobs = true;
                                // mark where to end execution
                                jobLevelBreak = job.c.size();
                            } else {
                                // copy the job out of the queue
                                job = parent->jobs.back();
                                // change the job in the queue to make it available for another worker
                                popTop(parent->jobs.back().numbers, parent->jobs.back().vertices);
                                parent->jobs.back().estimatedMax = topNumber(parent->jobs.back().numbers) + parent->jobs.back().c.size();
                                // mark where to end execution
                                jobLevelBreak = job.c.size()+1;
                            }
                            // mark that another job has been activated
                            ++parent->activeJobs;
                            c = job.c;
                        }
                        TRACE("threadFunc: calling expand", TRACE_MASK_THREAD, 1);
                        // work on the job
                        {
                            TRACEVAR(jobLevelBreak, TRACE_MASK_THREAD, 1);
                            // dig in
                            expand(job);
                            ++jobsDone;
                        }
                        TRACE("threadFunc: job completed", TRACE_MASK_THREAD, 1);
                        // deacivate the completed job
                        {
                            std::lock_guard<std::mutex> lk(parent->mutexJobs);
                            --parent->activeJobs;
                        }
                    }
                } // timer scope end           
                // send statistics to the parent
                {
                    TRACE("threadFunc: reposting stats", TRACE_MASK_THREAD, 1);
                    std::lock_guard<std::mutex> lk(parent->mutexJobs); 
                    parent->workerActiveTimes.push_back(timer.totalSeconds());
                    parent->workerSteps.push_back(steps);
                }
            } catch (const char* e) {
                std::cout << "Exception in thread: " << e << "\n";
            }
        }
        
        // main recursive function (parallel)
        void expand(Job& job) {
            ++steps;
            TRACE("expand start", TRACE_MASK_CLIQUE, 2);
            
            localMaxSize = parent->maxSize;
            TRACEVAR(localMaxSize, TRACE_MASK_CLIQUE, 2);
            while (notEmpty(job.numbers)) {
                if (job.estimatedMax <= localMaxSize || parent->killTimer.timedOut) {return;}
                Job newJob;
                auto v = topVertex(job.numbers, job.vertices);
                popTop(job.numbers, job.vertices);
                job.estimatedMax = c.size() + topNumber(job.numbers);
                graph->intersectWithNeighbours(v, job.vertices, newJob.vertices);
                
                c.add(v);
                if (c.size() > localMaxSize) { // condition (newJob.vertices.size() == 0) is left out to make maxSize up to date at all times
                    localMaxSize = parent->saveSolution(c);
                }
                
                TRACEVAR(newJob.vertices.size(), TRACE_MASK_CLIQUE, 2);
                if (newJob.vertices.size() > 0) {
                    // number vertices
                    numberSort(c, newJob.vertices, newJob.numbers, localMaxSize);
                    newJob.estimatedMax = c.size() + topNumber(newJob.numbers);
                    TRACEVAR(newJob.estimatedMax, TRACE_MASK_CLIQUE, 2);
                    
                    // continue exploration of the search tree within this job
                    TRACE("expand calling expand", TRACE_MASK_CLIQUE, 2);
                    expand(newJob);
                    // <= instead of <, because c.remove(v) has not been called yet
                    if (c.size() <= jobLevelBreak) {
                        TRACE("breaking worker search tree", TRACE_MASK_CLIQUE, 2);
                        TRACEVAR(jobLevelBreak, TRACE_MASK_CLIQUE, 2); 
                        return;
                    }
                }
                c.remove(v);
            }
            TRACE("expand end", TRACE_MASK_CLIQUE, 2);
        }
    };

    
protected:
    using InitialSorter::initialSort;
        
    std::string algorithmName;
    Graph* graph;
    VertexId n;                         // number of vertices
    unsigned int maxSize;               // size of max clique
    unsigned int numThreads;            // stores the number of threads used in the last search (where this number a parameter to the function)
    VertexSet maxClique;
    std::deque<Job> jobs;
    bool requireMoreJobs;
    size_t maxJobs, activeJobs;
    PrecisionTimer timer;
    std::vector<double> workerActiveTimes;
    std::vector<unsigned long long> workerSteps;
    std::mutex mutexJobs, mutexQ;
    KillTimer killTimer;
    
public:
    VertexSet knownC;

    ParallelMaximumCliqueProblem(Graph& graph) : graph(&graph), n(graph.getNumVertices()) {}
    
    // get the result of the search - maximal clique of the provided graph
    const VertexSet& getClique() const {return maxClique;}
    
    // output statistics of the last search (mostly timer readings); colored=true produces colored text on terminals supporting ANSI escape sequences
    void outputStatistics(bool colored = true) {
        std::ostringstream algorithmName;
        auto basefmt = std::cout.flags();
        auto baseFill = std::cout.fill();
        algorithmName << "pMC[" << numThreads << " threads, " << maxJobs << " jobs](" << ClassName<VertexId>::getValue() << ","
            << ClassName<VertexSet>::getValue() << "," << ClassName<Graph>::getValue() << "," << ClassName<InitialSorter>::getValue() << ") ";
        std::cout << "-- " << std::setw(80-3) << std::setfill('-') << std::left << algorithmName.str();
        std::cout.flags(basefmt);
        std::cout << std::setfill(baseFill) << "\n";
        unsigned long long steps = 0;
        for (size_t i = 0; i < workerSteps.size(); ++i) {
            steps += workerSteps[i];
        }
        std::cout << "search took " << timer.totalSeconds() << "s; " << steps << " steps\n";
        VertexSet c = getClique();
        graph->remap(c);
        if (wasSearchInterrupted()) std::cout << "Warning, search has been interrupted, the results might not be correct\n";
        std::cout << "Clique (" << getClique().size() << "): " << c;
        std::cout.flags(basefmt);
        std::cout << std::setfill(baseFill);
    }
    
    double workerEfficiency() const {
        double maxTime = timer.totalSeconds();
        double eff = 0;
        for (size_t i = 0; i < workerActiveTimes.size(); ++i) {
            eff += workerActiveTimes[i];
        }
        return eff / (maxTime*workerActiveTimes.size());
    }
    
    void sortLastJob() {
        if (jobs.size() > 1) {
            for (auto j = jobs.end()-1; (j != jobs.begin()) && ((j-1)->vertices.size() > j->vertices.size()); --j)
                swap(*j, *(j-1));
        }
    }
    
    void addJob(Job& job) {
        // LIFO queue, less copying
        jobs.push_front(Job());
        swap(jobs.front(), job);
        requireMoreJobs = jobs.size() == maxJobs;
        
        /*
        // Jobs queue is of LIFO type
        jobs.push_front(job);
        requireMoreJobs = jobs.size() == maxJobs;
        */
        
        // Job sorting, it works better without one :(
        //jobs.push_back(job);
        //sortLastJob();
    }
    
    void swapJob(Job& job) {
        if (jobs.size() > 0 && job.estimatedMax < jobs.back().estimatedMax) {
            //swap(job, jobs.back());
            //sortLastJob();
        }
        /* // DEBUG
        static int cnt = 1;
        if (cnt > 0) {
            --cnt;
            for (auto& j: jobs)
                std::cout << j.estimatedMax << " ";
            std::cout << "\n";
        }*/
    }
    
    // run the search for max clique
    void search(unsigned int numThreads, unsigned int numJobs, std::vector<int>& affinities) {
        killTimer.start(10000);
        ScopeTimer t(timer);
        this->numThreads = numThreads;
        
        VertexSet c; // clique
        VertexSet p; // working set of vertices
        NumberedSet numbers;       // ordered set of colors
        {
            // setting the order of vertices (initial sort that renumbers vertices in the input graph)
            c.reserve(n);
            p.reserve(n);
            for (VertexId i = 0; i < n; i++) 
                p.add(i);
            
            InitialSorter::init(graph);
            TRACE(typeid(InitialSorter).name(), TRACE_MASK_CLIQUE, 1);
            initialSort(c, p, numbers);
            
            // some initial sorts (e.g. MCR) also find a clique
            maxSize = c.size();
            if (maxSize > 0)
                saveSolution(c);
        }
        if (numbers.size() == 0) {
            // if initial sort did not setup "numbers", numberSort must be called
            numberSort(c, p, numbers, maxSize);
        }
        
        c.clear();
        // Generate root job
        maxJobs = std::min(numJobs, numThreads*1000);
        //jobs.reserve(maxJobs+1);
        jobs.push_back(Job(c, p, numbers, c.size() + topNumber(numbers)));
        activeJobs = 0;
        requireMoreJobs = jobs.size() < maxJobs;
        
        // Create threads and workers
        std::vector<std::unique_ptr<std::thread> > threads;
        std::vector<Worker> workers;
        threads.resize(numThreads);
        workers.resize(numThreads);
        
        // associate thread & worker pairs
        for (unsigned int i = 0; i < numThreads; ++i) {
            workers[i].setup(this, i);
            threads[i] = std::unique_ptr<std::thread>(new std::thread([&workers, i](){workers[i].threadFunc();}));
            if (affinities.size() > 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(affinities[i % affinities.size()], &cpuset);
                pthread_setaffinity_np(threads[i]->native_handle(), sizeof(cpu_set_t), &cpuset);
            }
        }
        // wait for all the workers to finish
        for (unsigned int i = 0; i < numThreads; ++i) {
            threads[i]->join();
        }
        killTimer.cancel();
    }
    
    bool wasSearchInterrupted() const {return killTimer.timedOut;}
    
protected:
    // when a clique, larger than its predecessor is found, call this function to store it
    unsigned int saveSolution(const VertexSet& c) {
        unsigned int ret;
        // make a copy of clique
        TRACE("Saving solution", TRACE_MASK_CLIQUE, 2);
        TRACEVAR(c.size(), TRACE_MASK_CLIQUE, 2);
        {
            std::lock_guard<std::mutex> lk(mutexQ); 
            if (maxSize < c.size()) {
                maxSize = c.size();
                maxClique = c;
                //std::cout << "new clique " << maxSize << "\n";
            }
            ret = maxSize;
        }
        
        return ret;
    }
};


#endif // header guard 
