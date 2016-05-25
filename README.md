This is a fork of Manzil Zaheer's multithreaded [CoverTree](https://github.com/manzilzaheer/CoverTree) implementation, with modifications to store feature-ids as well as features & an embedded [Mongoose](https://github.com/cesanta/mongoose) asynchronous HTTP server to enable queries to be made over HTTP.

To use, 

1. Modify `config.json` so that the filepaths point to your list of filenames and your file of features, i.e
```
{"19" : [
            {"region" : "<Name of this dataset>",
             "data" : "<Filepath of data>",
             "filenames" : "<List of filenames that correspond to the data>"},
    
            {"region" : "<Another dataset>",
             "data" : "<Filepath of data>",
             "filenames" : "<List of filenames that correspond to the data>"}
        ]
}
```

Currently, the datafile format is `<int number of points><int number of dimensions><array of doubles of point data>`. The file [data/convert.py](data/convert.py) converts a .t7 file that contains an array of feature-vectors into this format. You could also modify the [read_point_file](src/cover_tree/main.cpp) function to read your own data format, or trivially modify `convert.py` to convert a `.npy` to this format.

2. `make` (if not already compiled)
3. Run `dist/cover_tree <path to config.json> <port number to use for server>`

The server responds to queries of the form:

`<url of server>:<port>/?filename=<filename of feature>&limit=<number results to return>&level=19&region=<dataset name>`

## Requirements
1. C++14 compiler
2. OpenMP
3. Optional: `torchfile` to use [convert.py](data/convert.py)

## Libraries used

* [Mongoose](https://github.com/cesanta/mongoose)
* [Picojson](https://github.com/kazuho/picojson)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

