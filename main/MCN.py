import numpy as np
import matplotlib.pyplot as plt
import main.utils as utils
import time

# ---------------------------- 说明 ----------------------------------
# MCN的python复现
# ---------------------------- 说明 ----------------------------------


class MCNParams:
    """
    a struct define the input params MCN class use
    """
    def __init__(self, probAddCon, nCellPerCol, nConPerCol,
                 minColActivity, nColPerPattern, kActiveCol):
        self.probAddCon = probAddCon
        self.nCellPerCol = nCellPerCol
        self.nConPerCol = nConPerCol
        self.minColActivity = minColActivity
        self.nColPerPattern = nColPerPattern
        self.kActiveCol = kActiveCol


class MCN:

    def __init__(self, params):

        # MCNParams class define the params
        self.params = params

        self.nCols = 0
        self.winnerCells = []
        self.prevWinnerCells = []

        self.FF = np.empty((self.params.nConPerCol, self.nCols), dtype=np.int)
        self.P = np.empty((self.params.nCellPerCol, self.nCols), dtype=np.bool)
        self.prevP = np.empty_like(self.P, dtype=np.bool)

        self.burstedCol = np.empty((self.nCols, ), dtype=np.bool)
        self.predicitionConnections = []

    def prepareNewIteration(self):

        # winnerCells and P need to reset each time
        self.prevWinnerCells = self.winnerCells
        self.prevP = self.P

        self.winnerCells = []
        if self.nCols > 0:
            self.P = np.zeros_like(self.P)
            self.burstedCol = np.zeros_like(self.burstedCol)


    def resetPredP(self):
        self.prevP = np.empty((self.params.nCellPerCol, self.nCols), dtype=np.bool)

    def createNewColumn(self, inputSDR, nNewColumn):

        nonZeroIdx = np.where(inputSDR > 0)[0]

        start_id = self.nCols
        for i in range(nNewColumn):
            self.nCols += 1

            sampleIdx = np.random.randint(0, len(nonZeroIdx), self.params.nConPerCol)
            tmp = nonZeroIdx[sampleIdx].reshape((-1, 1))
            self.FF = np.concatenate((self.FF, tmp), axis=1)

            newPcol = np.zeros((self.params.nCellPerCol, 1), dtype=np.bool)
            self.P = np.concatenate((self.P, newPcol), axis=1)
            self.prevP = np.concatenate((self.prevP, newPcol), axis=1)
            self.burstedCol = np.concatenate((self.burstedCol, np.array([0], dtype=bool)))
        for k in range(nNewColumn * self.params.nCellPerCol):
            self.predicitionConnections.append([])

        return np.arange(start_id, self.nCols)


    def compute(self, inputSDR, supressLearningFlag):
        """
        compute sequence descriptor
        :param inputSDR:
        :param supressLearningFlag: in case of inference, not learning
        :return:
        """

        self.prepareNewIteration()

        # compare SDR with minicolumn
        simScore = np.sum(inputSDR[self.FF], axis=0) / self.params.nConPerCol
        sort_idx = np.argsort(simScore)
        topk_sort_idx = sort_idx[-self.params.kActiveCol:]
        topk_sort_score = simScore[topk_sort_idx]
        if not supressLearningFlag:
            # if all activities below threshold, then create a new
            # activity and make it active
            # otherwise select the top k most active ones
            if len(simScore):
                activeCols = topk_sort_idx[topk_sort_score > self.params.minColActivity]
                # activeCols = np.array(self.getActiveCols(simScore, supressLearningFlag), dtype=np.int)
            else:
                activeCols = np.empty((0, ), dtype=np.int)

            activeCols = np.concatenate((activeCols, self.createNewColumn(inputSDR, max(0, self.params.nColPerPattern - len(activeCols)))))

        else:
            # in non-learning mode, take the k most active columns
            # activeCols = np.array(self.getActiveCols(simScore, supressLearningFlag), dtype=np.int)
            activeCols = topk_sort_idx
            # if len(activeCols) == 0:
            #     sort_idx = np.argsort(simScore)
            #     activeCols = sort_idx[-self.params.nColPerPattern:]

        for eachActiveCol in activeCols:
            predictedIdx = np.where(self.prevP[:, eachActiveCol] > 0)[0]

            if len(predictedIdx):
                for each_predictedIdx in predictedIdx:
                    self.activatePredictions(eachActiveCol, each_predictedIdx)
                    self.winnerCells.append(eachActiveCol * self.params.nCellPerCol + each_predictedIdx)
            else:
                winnerCell = self.burst(eachActiveCol, supressLearningFlag)
                for each in winnerCell:
                    self.winnerCells.append(eachActiveCol * self.params.nCellPerCol + each)

        if not supressLearningFlag:
            self.learnPreditions()
            # predict newly learned preditions, i think it's useless
            for colIdx in range(self.nCols):
                if self.burstedCol[colIdx]:
                    for i in range(self.params.nCellPerCol):
                        self.activatePredictions(colIdx, i)

        return self.winnerCells

    def activatePredictions(self, colIdx, cellIdx):
        predIdx = self.predicitionConnections[colIdx * self.params.nCellPerCol + cellIdx]
        for each in predIdx:
            c = each // self.params.nCellPerCol
            r = each % self.params.nCellPerCol
            self.P[r, c] = True

    def burst(self, colIdx, supressLearningFlag):

        self.burstedCol[colIdx] = True
        for i in range(self.params.nCellPerCol):
            self.activatePredictions(colIdx, i)

        # winnerCell is the cells with fewest connections with other cells
        st = colIdx * self.params.nCellPerCol
        nCon = []
        for i in range(self.params.nCellPerCol):
            nCon.append(len(self.predicitionConnections[st + i]))

        if not supressLearningFlag:
            # inhibit winning cells from the last iteration
            for i in self.prevWinnerCells:
                col = i // self.params.nCellPerCol
                if col == colIdx:
                    nCon[i % self.params.nCellPerCol] += self.params.nCellPerCol

            # find the fewest ones
            candidateIdx = [0]
            minV = nCon[0]
            for i in range(1, len(nCon)):
                if nCon[i] < minV:
                    candidateIdx = [i]
                    minV = nCon[i]
                elif nCon[i] == minV:
                    candidateIdx.append(i)

            nCan = len(candidateIdx)

            if nCan == 1:
                return [candidateIdx[0]]
            else:
                chosenIdx = np.random.randint(0, nCan, 1)
                return [candidateIdx[chosenIdx[0]]]

        else:
            # in case of inference, return all used winner cells
            winnerIdx = np.where(np.array(nCon) > 0)[0]
            if len(winnerIdx):
                return winnerIdx

            return [np.random.randint(0, self.params.nCellPerCol, 1)[0]]


    def learnPreditions(self):

        for prevIdx in self.prevWinnerCells:
            prevIdxCol = prevIdx // self.params.nCellPerCol
            for curIdx in self.winnerCells:
                curIdxCol = curIdx // self.params.nCellPerCol
                if prevIdxCol == curIdxCol:
                    continue

                existingPredConFlag = self.checkExistingPredCon(prevIdxCol, curIdx)
                if not existingPredConFlag or np.random.rand() <= self.params.probAddCon:
                    if curIdx not in self.predicitionConnections[prevIdx]:
                        self.predicitionConnections[prevIdx].append(curIdx)


    def checkExistingPredCon(self, prevColIdx, curCellIdx):
        st = prevColIdx * self.params.nCellPerCol
        for i in range(self.params.nCellPerCol):
            if curCellIdx in self.predicitionConnections[st + i]:
                return True

        return False


    def visualizeCon(self, displayCol=10):

        plt.figure()
        dis = 5
        dCol = displayCol
        plt.title('Prediction Connections')
        plt.xlim(0, dCol * dis)
        plt.ylim(0, self.params.nCellPerCol * dis)

        for k, con in enumerate(self.predicitionConnections):
            x = k // self.params.nCellPerCol * dis
            if x >= dCol * dis:
                break
            y = k % self.params.nCellPerCol
            y = (self.params.nCellPerCol - 1 - y) * dis
            plt.plot(x, y, 'o', color='blue')
            if len(con):
                for each in con:
                    cx = each // self.params.nCellPerCol * dis
                    cy = each % self.params.nCellPerCol
                    cy = (self.params.nCellPerCol - 1 - cy) * dis
                    plt.plot([x, cx], [y, cy], '-', color='red')



def getSim(w1, w2):
    """

    :param w1: winner cell which should be a list
    :param w2:
    :return: simularity score
    """
    w1 = set(w1)
    w2 = set(w2)
    return len(w1 & w2) / len(w1 | w2)


def runMCN(params, dbFeat, qFeat, gt):

    # st = time.time()
    _, old_dims = dbFeat.shape
    new_dims = 8192
    P = np.random.rand(old_dims, new_dims // 2)
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    D1_slsbh = utils.getLSBH(dbFeat, P, 0.25)
    D2_slsbh = utils.getLSBH(qFeat, P, 0.25)

    mcn = MCN(params)
    train_winnerCells = []
    for i in range(D1_slsbh.shape[0]):
        train_winnerCells.append(mcn.compute(D1_slsbh[i, :], False))

    valid_winnerCells = []
    mcn.resetPredP()
    for i in range(D2_slsbh.shape[0]):
        valid_winnerCells.append(mcn.compute(D2_slsbh[i, :], True))

    # print('Done! cost : %.3f' % (time.time() - st))
    # get similarity matrix
    S_mcn = np.zeros((dbFeat.shape[0], qFeat.shape[0]))
    for k1, each_v in enumerate(valid_winnerCells):
        for k2, each_t in enumerate(train_winnerCells):
            S_mcn[k2, k1] = getSim(each_v, each_t)
    # time_cost = time.time() - st
    # P, R = utils.drawPR(S_mcn, gt)
    # ap = utils.calAvgPred(P, R)
    del train_winnerCells, valid_winnerCells, mcn

    return S_mcn

def runMCN_SDR(params, dbFeat, qFeat, gt):

    mcn = MCN(params)
    train_winnerCells = []
    for i in range(dbFeat.shape[0]):
        train_winnerCells.append(mcn.compute(dbFeat[i, :], False))

    valid_winnerCells = []
    mcn.resetPredP()
    for i in range(qFeat.shape[0]):
        valid_winnerCells.append(mcn.compute(qFeat[i, :], True))

    # print('Done! cost : %.3f' % (time.time() - st))
    # get similarity matrix
    S_mcn = np.zeros((dbFeat.shape[0], qFeat.shape[0]))
    for k1, each_v in enumerate(valid_winnerCells):
        for k2, each_t in enumerate(train_winnerCells):
            S_mcn[k2, k1] = getSim(each_v, each_t)
    # time_cost = time.time() - st
    # P, R = utils.drawPR(S_mcn, gt)
    # ap = utils.calAvgPred(P, R)
    del train_winnerCells, valid_winnerCells, mcn

    return S_mcn
