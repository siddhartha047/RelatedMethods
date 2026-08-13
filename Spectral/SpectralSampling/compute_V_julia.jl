
using Pkg
Pkg.add("DelimitedFiles")
Pkg.add("NPZ")
Pkg.add("PyCall")
Pkg.add("Laplacians")
Pkg.add("Conda")

using PyCall
using DelimitedFiles
using SparseArrays
using Laplacians
using LinearAlgebra
using Conda
Conda.add("scipy")


np = pyimport("numpy");
scipy = pyimport("scipy");

const scipy_sparse_find = pyimport("scipy.sparse")["find"]
function mysparse(Apy::PyObject)
    IA, JA, SA = scipy_sparse_find(Apy)
    return sparse(Int[i+1 for i in IA], Int[i+1 for i in JA], SA)
end

function compute_V(a, JLfac=4.0)

  f = approxchol_lap(a,tol=1e-2);

  n = size(a,1)
  k = round(Int, JLfac*log(n)) # number of dims for JL
  U = wtedEdgeVertexMat(a)
  m = size(U,1)
  R = randn(Float64, m,k)
  UR = U'*R;
  V = zeros(n,k)
  for i in 1:k
    V[:,i] = f(UR[:,i])
  end
  return V
end


#filenames = ["Cora","Citeseer","Pubmed","Phy","CS","Photo","Computers"]
#filenames=["Cora","Pubmed","Phy","CS"]
#filenames=["Photo","Computers"]
#filenames=["git","twitch_DE","twitch_FR","wiki_crocs","wiki_squirrels"]
#filenames=["reddit_adj_symm"]

#filenames=["Pubmed","Phy","CS"]
# define the path to the adjacency matrix files in the next line and uncomment it

filepath= "Amazon-ratings.npz"

filename = "Amazon-ratings"

str = string(filepath)
println("Loading file:... ")
data = scipy.sparse.load_npz(filepath);
println("Finished loading file, ",str)


println("Converting into Julia format")
B = mysparse(data);
B = convert(SparseMatrixCSC{Float64,Int64},B);
println("Finished converting into Julia format")

println("Matrix dimensions: ", size(B))
if size(B, 1) != size(B, 2)
    error("Adjacency matrix is not square!")
end


println("Computing V matrix..")
V = @time compute_V(B);
println("Finished computing V matrix..")

println("Dimensions of V are: ", size(V))

println("saving V matrix:...")
filename=string("V_",filename,".csv");
writedlm(filename,  V, ',')