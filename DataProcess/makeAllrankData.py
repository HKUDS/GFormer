from Params import *
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import pickle

class DataHandler:
	def __init__(self):
		if args.data == 'gowalla':
			predir = './sparse_gowalla/'
		elif args.data == 'yelp':
			predir = './sparse_yelp/'
		elif args.data == 'amazon':
			predir = './sparse_amazon/'
		else:
			predir = './%s/' % args.data
		self.predir = predir
		self.trnfile = predir + 'trn_mat'
		self.tstfile = predir + 'tst_int'

	def LoadData(self):
		with open(self.trnfile, 'rb') as fs:
			trnMat = (pickle.load(fs) != 0).astype(np.float32)
		# test set
		with open(self.tstfile, 'rb') as fs:
			tstLst = np.array(pickle.load(fs))
		tstStat = (tstLst != None)
		tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
		self.trnMat = trnMat
		self.tstLst = tstLst
		self.tstUsrs = tstUsrs
		args.user, args.item = self.trnMat.shape


handler = DataHandler()
handler.LoadData()

trnMat = coo_matrix(handler.trnMat)
tstLst = handler.tstLst
row = list(trnMat.row)
col = list(trnMat.col)
data = list(trnMat.data)
for i in range(args.user):
	if tstLst[i] is not None:
		row.append(i)
		col.append(tstLst[i])
		data.append(1)

row = np.array(row)
col = np.array(col)
data = np.array(data)

leng = len(row)
indices = np.random.permutation(leng)
trn = int(leng * 0.7)
val = int(leng * 0.75)

trnIndices = indices[:trn]
trnMat = coo_matrix((data[trnIndices], (row[trnIndices], col[trnIndices])), shape=[args.user, args.item])

valIndices = indices[trn:val]
valMat = coo_matrix((data[valIndices], (row[valIndices], col[valIndices])), shape=[args.user, args.item])

tstIndices = indices[val:]
tstMat = coo_matrix((data[tstIndices], (row[tstIndices], col[tstIndices])), shape=[args.user, args.item])

num = np.sum(trnMat != 0) + np.sum(valMat != 0) + np.sum(tstMat != 0)
print(num, num / (trnMat.shape[0] * trnMat.shape[1]))

with open('./%s/trnMat.pkl' % handler.predir, 'wb') as fs:
	pickle.dump(trnMat, fs)
with open('./%s/valMat.pkl' % handler.predir, 'wb') as fs:
	pickle.dump(valMat, fs)
with open('./%s/tstMat.pkl' % handler.predir, 'wb') as fs:
	pickle.dump(tstMat, fs)