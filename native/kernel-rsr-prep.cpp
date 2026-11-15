#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct int4 { int32_t x,y,z,w; };
static int4 make_int4(int x,int y,int z,int w) { return {x,y,z,w}; }

template<class T> static std::vector<T> read_values(const std::string & path, size_t count) {
    std::vector<T> result(count);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(result.data()), count * sizeof(T)))
        throw std::runtime_error("truncated input: " + path);
    if (input.peek() != EOF) throw std::runtime_error("oversized input: " + path);
    return result;
}
template<class T> static void save_values(const std::string & path, const std::vector<T> & values) {
    std::ofstream out(path,std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(T))) throw std::runtime_error("index write failed");
}

static void prepare_rsr_index(const std::string & prefix,int m,int k,int block) {
    if (m <= 0 || k <= 0 || k % 32 || k > 65535 || (block != 4 && block != 8 && block != 12)) throw std::runtime_error("invalid RSR geometry");
    const auto weights=read_values<uint32_t>(prefix+".weights",size_t(m)*k/32);
    const int blocks=(m+block-1)/block;
    std::vector<uint16_t> permutations(size_t(blocks)*k);
    std::vector<int32_t> metadata;
    std::vector<int4> groups;
    for(int b=0;b<blocks;++b) {
        std::vector<int> counts(1<<block), patterns(k), offsets((1<<block)+1);
        for(int j=0;j<k;++j) {
            int mask=0;
            for(int r=0;r<block;++r) {
                const int row=b*block+r;
                mask=(mask<<1) | (row<m ? int((weights[size_t(row)*(k/32)+j/32]>>(j%32))&1u) : 0);
            }
            patterns[j]=mask;
            ++counts[mask];
        }
        metadata.push_back(int32_t(groups.size()));
        int nonempty=0;
        for(int mask=0;mask<(1<<block);++mask) {
            offsets[mask+1]=offsets[mask]+counts[mask];
            if(counts[mask]) {
                int rows=0;
                for(int r=0;r<block;++r) rows |= ((mask>>(block-1-r))&1)<<r;
                groups.push_back(make_int4(offsets[mask],offsets[mask+1],rows,0));
                ++nonempty;
            }
        }
        metadata.push_back(nonempty);
        auto cursor=offsets;
        for(int j=0;j<k;++j) permutations[size_t(b)*k+cursor[patterns[j]]++]=uint16_t(j);
    }
    const auto name=prefix+".rsr"+std::to_string(block);
    save_values(name+".perms",permutations);
    save_values(name+".groups",groups);
    save_values(name+".meta",metadata);
    save_values(name+".sizes",std::vector<uint32_t>{uint32_t(groups.size())});
    printf("index,%zu,%zu,%d\n",permutations.size()*sizeof(uint16_t)+groups.size()*sizeof(int4)+metadata.size()*sizeof(int32_t),groups.size(),blocks*block);
}


int main(int argc,char ** argv) {
    try {
        if(argc!=5) throw std::runtime_error("prefix rows columns block required");
        prepare_rsr_index(argv[1],std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));
        return 0;
    } catch(const std::exception & error) {
        fprintf(stderr,"%s\n",error.what());
        return 2;
    }
}
