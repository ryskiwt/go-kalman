package kalman

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// Filter filters input mat64.Dense
func (f *Filter) Filter(s *mat64.Dense) *mat64.Dense {

	rH, _ := f.H.Dims()
	rS, cS := s.Dims()
	ret := mat64.NewDense(rH, cS, nil)

	var m0, m1, m2 mat64.Dense
	var di0, di, k, kh mat64.Dense
	var ke, e mat64.Vector
	var retVec mat64.Vector
	var retAry []float64

	FT := f.F.T()
	GT := f.G.T()
	HT := f.H.T()

	for j := 0; j < cS; j++ {

		// x = F @ x
		f.X.MulVec(f.F, f.X)

		// V = F @ V @ F.T + G @ Q @ G.T
		m0.Mul(f.F, f.V)
		f.V.Mul(&m0, FT)

		m1.Mul(f.G, f.Q)
		m2.Mul(&m1, GT)
		f.V.Add(f.V, &m2)

		// d = (H @ V @ H.T + R)^-1
		di0.Mul(f.H, f.V)
		di.Mul(&di0, HT)
		di.Add(&di, f.R)
		di.Inverse(&di)

		// V @ H.T @ d^-1
		k.Mul(f.V, HT)
		k.Mul(&k, &di)

		// e = y - H @ x
		y := s.ColView(j)
		e.MulVec(f.H, f.X)
		e.SubVec(y, &e)

		// NaN to 0
		allNaN := true
		for i := 0; i < rS; i++ {
			v := y.At(i, 0)
			if math.IsNaN(v) {
				y.SetVec(i, 0)
				allNaN = false
			}
		}

		if allNaN {
			// x = x + K @ e
			ke.MulVec(&k, &e)
			f.X.AddVec(f.X, &ke)

			// V = (I - K @ H) @ V
			kh.Mul(&k, f.H)
			kh.Sub(f.I, &kh)
			f.V.Mul(&kh, f.V)
		}

		retVec.MulVec(f.H, f.X)
		retAry = mat64.Col(retAry, 0, &retVec)
		ret.SetCol(j, retAry)
	}

	return ret
}
