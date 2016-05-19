# ifndef PARALLEL_COVER_TREE_H
# define PARALLEL_COVER_TREE_H

#include <iostream>
#include <random>
#include <future>
#include <mutex>
#include <thread>
#include <chrono>

#include "cover_tree.h"

/* Parallel implementation of cover tree using work-stealing fork-join */

class ParallelMake
{
	int left;
	int right;
    std::vector<point>& pList;
	
	CoverTree* CT;

	void run()
	{
		std::cout << "Inserting 50000..." << std::endl;
        CT = new CoverTree(pList, left, right);
        std::cout << "Inserted 50000! " << std::endl;
	}

public:
    ParallelMake(int left, int right, std::vector<point>& pL) : pList(pL)
	{
		this->left = left;
		this->right = right;
		this->CT = NULL;
	}
	~ParallelMake()
	{	}

	int compute()
	{
		if (right - left < 50000)
		{
			run();
			return 0;
		}

		int split = (right - left) / 2;

		ParallelMake* t1 = new ParallelMake(left, left + split, pList);
		ParallelMake* t2 = new ParallelMake(left + split, right, pList);

		std::future<int> f1 = std::async(std::launch::async, &ParallelMake::compute, t1);
		std::future<int> f2 = std::async(std::launch::async, &ParallelMake::compute, t2);

		f1.get();
		f2.get();
        
        ts = std::chrono::high_resolution_clock::now();
		if (t1->CT->get_level() > t1->CT->get_level())
		{
			std::cout << "merging t1->t2" << std::endl;
            t1->CT->Merge(t2->CT);
			CT = t1->CT;
		}
		else
		{
            std::cout << "merging t2->t1" << std::endl;
            t2->CT->Merge(t1->CT);
			CT = t2->CT;
		}
        tn = std::chrono::high_resolution_clock::now();
        std::cout << "merge time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << " ";
        
        ts = std::chrono::high_resolution_clock::now();
		delete t1;
		delete t2;
        tn = std::chrono::high_resolution_clock::now();

        std::cout << "del time: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;

		return 0;
	}

	CoverTree* get_result()
	{
		return CT;
	}

};


#endif
