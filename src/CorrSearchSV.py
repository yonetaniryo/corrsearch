import glob
import sys
import numpy as np
import cv2
from scipy.signal import medfilt
from sklearn.decomposition import PCA
import json
import argparse
from scipy.stats import scoreatpercentile
from skimage.segmentation import mark_boundaries
from skimage.measure import label as bwlabel


def grabcut(img, targetness):
    u"""
    Segmenting the best target-like region from an targetness map.
    """

    mask = np.ones(img.shape[:2], np.uint8) * cv2.GC_BGD
    score_th = scoreatpercentile(targetness, 95)
    mask[targetness >= score_th] = cv2.GC_PR_FGD
    score_th = scoreatpercentile(targetness, 99)
    mask[targetness >= score_th] = cv2.GC_FGD
    mask = cv2.medianBlur(mask, 15)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    lab_mask2 = bwlabel(mask2)
    lab_list = np.unique(lab_mask2.flatten())[1:]
    lab_argmax = np.argmax(
        [np.max(targetness[lab_mask2 == i]) for i in lab_list])
    mask2[lab_mask2 != lab_list[lab_argmax]] = 0
    img2 = img.copy()
    img2[mask2 < 1, :] = [0, 43, 54]
    img2 = mark_boundaries(img2, mask2)

    return img2, mask2


def nanmean(mat, axis=1):
    mmat = np.ma.masked_array(np.array(mat), np.isnan(mat))
    mm = np.mean(mmat, axis=axis)
    mm = np.matrix(mm)
    if(axis == 1):
        return mm.T
    else:
        return mm


class CSBASE():
    u"""
    Base class of CorrSearch
    """

    def __init__(self, mode, imgsize, interval, step, gstep, nframes, ext):
        self.mode = mode
        self.imgsize = imgsize
        self.interval = interval
        self.step = step
        self.gstep = gstep
        self.nframes = nframes
        self.ext = ext

    def localize_target(self, tardir, obsdir, svdir, clsf, scaler):
        u"""
        Pipeline function to localize target instances in an observer video.
        Output (fullmap) is a targetness map in the form of [N x M x T] numpy
        array where T = self.nframes / self.gstep.
        """

        nframes = self.nframes
        gstep = self.gstep
        ext = self.ext

        score_h = self.calc_correlation(tardir, obsdir, svdir)
        svfiles = [sorted(glob.glob('%s/*%s' % (x, ext))) for x in svdir]
        sv_h = [self.__load_supervoxel(x[0:nframes:gstep]) for x in svfiles]

        scmap_h, feat_h = self.__extract_feat(sv_h, score_h)

        fullmap = self.estimate_targetness(scmap_h, feat_h, clsf, scaler)

        return fullmap

    def calc_correlation(self, tardir, obsdir, svdir):
        u"""
        Evaluating correlation-based targetness.
        """

        interval = self.interval
        step = self.step
        ext = self.ext
        nframes = self.nframes

        n_seq = nframes / interval
        n_scales = len(svdir)
        score_h = [[[] for seq in range(n_seq)] for h in range(n_scales)]
        tarfiles = sorted(glob.glob('%s/*%s' % (tardir, ext)))
        obsfiles = sorted(glob.glob('%s/*%s' % (obsdir, ext)))
        svfiles = [sorted(glob.glob('%s/*%s' % (x, ext))) for x in svdir]

        # Main process
        for seq in range(n_seq):

            print "seq #%d / %d" % (seq + 1, n_seq)
            start = seq * interval
            end = np.min(((seq + 1) * interval, nframes))

            print "Calculating motion...",
            tarimgs = [self.__load_img(x) for x in tarfiles[start:end:step]]
            gseq, _ = self.__extract_motion_pattern(tarimgs)
            obsimgs = [self.__load_img(x) for x in obsfiles[start:end:step]]
            _, lseq = self.__extract_motion_pattern(obsimgs)
            print "done."

            print "Matching motion...",
            for h in range(n_scales):
                print 'scale #%d/%d...' % (h + 1, n_scales),
                sv = self.__load_supervoxel(svfiles[h][start:end:step])
                score = self.__calc_zncc(sv, lseq, gseq)
                score_h[h][seq] = score

            print "done."

        return score_h

    def estimate_targetness(self, scmap_h, feat_h, clsf, scaler):
        u"""
        Estimating data-driven generic targetness.
        """

        n_scales = len(scmap_h)
        prior = [clsf.predict_proba(scaler.transform(feat_h[h])
                                    )[:, 1].reshape(scmap_h[h].shape)
                 for h in range(n_scales)]
        fullmap = np.prod(np.concatenate(
            [((prior[h]) * self.__sigm(scmap_h[h]))[..., np.newaxis]
             for h in range(n_scales)], 3), 3)

        return fullmap

    def __extract_feat(self, sv_h, score_h):
        scmap_h = []
        feat = []
        for h in range(len(sv_h)):
            sv = sv_h[h]
            score = score_h[h]
            score_sum = np.zeros_like(score[0])
            for t in range(len(score)):
                score[t][:, 0] *= score[t][:, 4]
                score[t][:, 2] *= score[t][:, 4]
                score[t][:, 3] *= score[t][:, 4]
                score[t][:, 5] *= score[t][:, 4]
                score_sum += score[t]
            # weighted sum by ovserved length of spvs
            score_sum[:, 0] /= score_sum[:, 4] + 1e-5
            score_sum[:, 2] /= score_sum[:, 4] + 1e-5
            score_sum[:, 3] /= score_sum[:, 4] + 1e-5
            score_sum[:, 5] /= score_sum[:, 4] + 1e-5
            scmap = score_sum[:, 0].take(sv.flatten()).reshape(sv.shape)
            lcov = np.sqrt(score_sum[:, 2].take(sv.flatten()
                                                ).reshape(sv.shape))
            size = score_sum[:, 3].take(sv.flatten()).reshape(sv.shape)
            length = score_sum[:, 4].take(sv.flatten()).reshape(sv.shape)
            lstd = score_sum[:, 5].take(sv.flatten()).reshape(sv.shape)

            scmap_h.append(scmap)

            feat.append(np.vstack((lcov.flatten(), size.flatten(),
                                   length.flatten(), lstd.flatten())).T)

        return scmap_h, feat

    def __extract_motion_pattern(self, imgs):

        gseq = []
        lseq = []
        for t in range(len(imgs)):
            tmp = self.__estimate_motion([imgs[np.max((0, t - 1))], imgs[t]])
            gseq.append(tmp[0])
            lseq.append(tmp[1])

        return gseq, lseq

    def __estimate_motion(self, imgs, win=5, eig_th=1e-4):

        img0 = imgs[0]
        img1 = imgs[1]
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        w, h = np.meshgrid(range(img0.shape[1]), range(img0.shape[0]))
        loc0 = (np.vstack((w.flatten(), h.flatten())).T).astype('float32')

        flow = cv2.calcOpticalFlowFarneback(img0_gray, img1_gray,
                                            0.5, 0, win, 3, 5, 1.2, 0)
        flow = (np.vstack((flow[:, :, 0].flatten(),
                           flow[:, :, 1].flatten())).T).astype('float32')

        # Removing irrelevant flows
        minEig = cv2.cornerMinEigenVal(img0_gray, blockSize=win * 3,
                                       borderType=cv2.BORDER_REPLICATE)
        loc0_of = loc0[minEig.flatten() > eig_th, :]
        loc1_of = flow[minEig.flatten() > eig_th, :] + loc0_of

        # Surf-based match
        loc0_sf, loc1_sf = self.__calc_surf(imgs)
        loc0_all = np.vstack((loc0_of, loc0_sf))
        loc1_all = np.vstack((loc1_of, loc1_sf))

        hom = cv2.findHomography(loc0_all, loc1_all, cv2.cv.CV_RANSAC, 1)[0]

        gm = cv2.perspectiveTransform(np.array([loc0]), hom)[0] - loc0
        lm = flow - gm
        gm = gm[:, 0] * 1j + gm[:, 1]
        lm = lm[:, 0] * 1j + lm[:, 1]
        lm[minEig.flatten() < eig_th] = 0

        return gm, lm

    def __calc_surf(self, imgs, param1=1500, param2=.1):

        knn = cv2.KNearest()
        surf = cv2.SURF(param1)
        kp0, ds0 = surf.detectAndCompute(imgs[0], None)
        kp1, ds1 = surf.detectAndCompute(imgs[1], None)

        sample = np.array(ds1)
        res = np.arange(len(kp1), dtype=np.float32)
        knn.train(sample, res)

        idx = []
        for h, d in enumerate(ds0):
            d = np.array(d, dtype=np.float32).reshape((1, 128))
            tmp = knn.find_nearest(d, 1)
            if(tmp[3][0] < param2):
                idx.append(tmp[1][0][0])
            else:
                idx.append(-1)

        loc1 = np.vstack([np.array(k.pt) for k in kp1])
        loc0 = np.vstack([np.array(k.pt) for k in kp0])
        idx = np.array(idx).flatten().astype('int')
        loc0 = loc0[idx != -1, :]
        idx = idx[idx != -1]
        loc1 = loc1[idx, :]

        return loc0, loc1

    def __calc_zncc(self, sv, lseq, gseq, LARGENUM=65536):

        gmean, lmean, size, length, lstd, id_list = \
            self.summarize_motion(sv, lseq, gseq)

        geval = np.matrix(gmean)
        leval = np.matrix(np.vstack(lmean))
        geval = np.matrix(np.ones(leval.shape[0])).T * geval

        geval[np.isnan(leval)] = np.nan
        g_ = geval - nanmean(geval, axis=1)
        l_ = leval - nanmean(leval, axis=1)
        gcov = nanmean(np.multiply(g_, g_), axis=1)
        lcov = nanmean(np.multiply(l_, l_), axis=1)
        glcov = nanmean(np.multiply(g_, l_), axis=1)
        score = np.divide(glcov, np.sqrt(np.multiply(gcov, lcov))+1e-5).T

        # Hack: ignoring small intervals
        th = self.interval / 2
        score[np.sum(~np.isnan(leval), 1).flatten() < th] = 0

        score_all = np.zeros((LARGENUM, 6))
        score_all[id_list, 0] = np.array(score).flatten()
        score_all[id_list, 1] = np.array(gcov).flatten()
        score_all[id_list, 2] = np.array(lcov).flatten()
        score_all[id_list, 3] = size
        score_all[id_list, 4] = length
        score_all[id_list, 5] = lstd

        return score_all

    def __load_img(self, imgfile):
        img = cv2.resize(cv2.imread(imgfile), tuple(self.imgsize),
                         interpolation=cv2.cv.CV_INTER_AREA)
        return img

    def __load_supervoxel(self, svfiles):
        sv = []
        for t in range(len(svfiles)):
            x = cv2.resize(cv2.imread(svfiles[t]).astype('float32'),
                           tuple(self.imgsize),
                           interpolation=cv2.cv.CV_INTER_NN).astype('int')
            sv.append(x[:, :, 0] + x[:, :, 1] * 255 + x[:, :, 2] * 255 * 255)
        sv = np.dstack(sv)

        return sv

    def __sigm(self, x):
        return 1. / (1. + np.exp(-x))


class CSPCA(CSBASE):
    u"""
    CorrSearch using a head-motion subspace cross correlation.
    This is the proposed method.
    """

    def summarize_motion(self, sv, lseq, gseq):
        lvol = [x.reshape((sv.shape[0], sv.shape[1])) for x in lseq]
        id_list = np.unique(sv.flatten())
        id_list = id_list[id_list != 0]

        gmean, pca = self.__calc_gm(gseq)
        tmp = [self.__calc_lm(np.array([lvol[t][sv[:, :, t] == spid]
                                        for t in range(len(lvol))]), pca)
               for spid in id_list]
        lmean = [x[0] for x in tmp]
        size = [x[1] for x in tmp]
        length = [x[2] for x in tmp]
        lstd = [x[3] for x in tmp]

        return gmean, lmean, size, length, lstd, id_list

    def __calc_gm(self, gseq, t_win=5):
        tmp = np.array([np.mean(x) for x in gseq])
        gmean = -1 * medfilt(tmp.real, t_win) + medfilt(tmp.imag, t_win) * 1j
        pca = PCA(n_components=1)
        gmean = pca.fit_transform(
            np.vstack((gmean.imag, gmean.real)).T).flatten()

        return gmean, pca

    def __calc_lm(self, lseq, pca, t_win=5):
        for x in range(len(lseq)):
            if(len(lseq[x]) == 0):
                lseq[x] = [np.nan]
        ltmp = np.array([np.mean(x) for x in lseq])
        ltmp = medfilt(ltmp.real, t_win) + medfilt(ltmp.imag, t_win) * 1j
        # applying PCA learned from global motions
        lmean_ = np.vstack((ltmp.imag, ltmp.real)).T
        idx = ~np.isnan(ltmp)
        lmean = np.zeros(ltmp.shape)
        if(any(idx)):
            lmean[idx] = pca.transform(lmean_[idx, :])
        lmean[~idx] = np.nan
        size = np.mean([len(x) for x in lseq])
        length = np.sum(idx)
        lstd = nanmean([np.std(x) for x in lseq], axis=0)[0, 0]

        return lmean, size, length, lstd


class CSABS(CSBASE):
    u"""
    Use this class if you wish to use an amplitude-based correlation instead of
    a head-motion subspace cross correlation.
    """

    def summarize_motion(self, sv, lseq, gseq):
        lvol = [x.reshape((sv.shape[0], sv.shape[1])) for x in lseq]
        id_list = np.unique(sv.flatten())
        id_list = id_list[id_list != 0]

        gmean = self.__calc_gm(gseq)
        tmp = [self.__calc_lm(np.array([lvol[t][sv[:, :, t] == spid]
                                        for t in range(len(lvol))]))
               for spid in id_list]
        lmean = [x[0] for x in tmp]
        size = [x[1] for x in tmp]
        length = [x[2] for x in tmp]
        lstd = [x[3] for x in tmp]

        return gmean, lmean, size, length, lstd, id_list

    def __calc_gm(self, gseq, t_win=5):
        tmp = np.array([np.mean(x) for x in gseq])
        gmean = -1 * medfilt(tmp.real, t_win) + medfilt(tmp.imag, t_win) * 1j
        gmean = np.abs(gmean)

        return gmean

    def __calc_lm(self, lseq, t_win=5):
        for x in range(len(lseq)):
            if(len(lseq[x]) == 0):
                lseq[x] = [np.nan]
        ltmp = np.array([np.mean(x) for x in lseq])
        ltmp = medfilt(ltmp.real, t_win) + medfilt(ltmp.imag, t_win) * 1j
        lmean = np.abs(ltmp)
        idx = ~np.isnan(ltmp)
        size = np.mean([len(x) for x in lseq])
        length = np.sum(idx)
        lstd = nanmean([np.std(x) for x in lseq], axis=0)[0, 0]

        return lmean, size, length, lstd


if(__name__ == '__main__'):

    argvs = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Correlation-based self search')
    parser.add_argument('-p', '--params', nargs=1, type=file,
                        metavar='params.json',
                        help='json file describing various parameters.')
    parser.add_argument('-t', '--tardir', nargs=1, type=str,
                        metavar='---/ppm',
                        help='directory of target video')
    parser.add_argument('-o', '--obsdir', nargs=1, type=str,
                        metavar='---/ppm',
                        help='directory of observer video')
    parser.add_argument('-s', '--svdir', nargs=1, type=str,
                        metavar='---/sv',
                        help='directory containing the subdirectories generated \
                        by gbh_stream.')
    parser.add_argument('-m', '--modelfile', nargs=1, type=str,
                        metavar='model.npy',
                        help='LDA and scaler to estimate generic targetness')
    parser.add_argument('-r', '--result', nargs=1, type=str,
                        metavar='result',
                        help='numpy file to output a localization result')

    p = parser.parse_args(argvs)
    params = json.load(p.params[0])
    if('pca' in params['mode']):
        print 'pca mode'
        C = CSPCA(**params)
    elif('abs' in params['mode']):
        print 'abs mode'
        C = CSABS(**params)

    tardir = p.tardir[0]
    obsdir = p.obsdir[0]
    svdir = sorted(glob.glob('%s/*' % p.svdir[0]))

    clsf, scaler = np.load(p.modelfile[0])
    fullmap = C.localize_target(tardir, obsdir, svdir, clsf, scaler)
    np.save(p.result[0], fullmap)
