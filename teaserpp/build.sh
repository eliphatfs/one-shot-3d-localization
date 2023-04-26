g++ -shared -o teaser.dll src/*.* pmc/*.cpp -DNDEBUG -I./include -I. -fopenmp -Wp,-w -O2
