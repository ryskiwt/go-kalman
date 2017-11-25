package kalman

import (
	"errors"

	"github.com/gonum/matrix/mat64"
)

// Config represents config for filter
type Config struct {
	// x = F@x + G@v, v ~ N(O,Q)
	F *mat64.Dense
	G *mat64.Dense
	Q *mat64.Dense

	// y = H@x + w, w ~ N(O,R)
	H *mat64.Dense
	R *mat64.Dense
}

// Filter represents kalman filter
type Filter struct {
	Config

	// Identical Matrix
	I *mat64.Dense

	// Internal States
	X *mat64.Vector
	V *mat64.Dense
}

// New creates a new kalman filter instance
func New(c *Config) (*Filter, error) {

	//
	// dimensions
	//

	rF, cF := c.F.Dims()
	rG, cG := c.G.Dims()
	rQ, cQ := c.Q.Dims()
	rH, cH := c.H.Dims()
	rR, cR := c.R.Dims()

	//
	// validate
	//

	if rF != cF {
		return nil, errors.New("F must be square matrix")
	}

	if rQ != cQ {
		return nil, errors.New("Q must be square matrix")
	}

	if rR != cR {
		return nil, errors.New("R must be square matrix")
	}

	if rF != rG {
		return nil, errors.New("row dim of F must be matched to row dim of G")
	}

	if cG != rQ {
		return nil, errors.New("column dim of G must be matched to row dim of Q")
	}

	if cH != cF {
		return nil, errors.New("column dim of H must be matched to column dim of F")
	}

	if rH != rR {
		return nil, errors.New("row dim of H must be matched to row dim of R")
	}

	// init internal states

	x := mat64.NewVector(cF, nil)
	v := mat64.NewDense(rF, cF, nil)
	ident := mat64.NewDense(rF, cF, nil)
	for i := 0; i < rF; i++ {
		ident.Set(i, i, 1.0)
	}

	return &Filter{
		Config: *c,
		I:      ident,
		X:      x,
		V:      v,
	}, nil
}

// Init initializes internal filter states
func (f *Filter) Init(x *mat64.Vector, v mat64.Matrix) error {

	rF, cF := f.F.Dims()

	//
	// X
	//

	if x == nil {
		f.X = mat64.NewVector(cF, nil)

	} else {
		rX, _ := x.Dims()
		if rX != cF {
			return errors.New("row dim of x must be matched to column dim of F")
		}
		f.X = x
	}

	//
	// V
	//

	if v == nil {
		var m *mat64.Dense
		m.Mul(f.G, f.Q)
		f.V.Mul(m, f.G.T())

	} else {
		rV, cV := v.Dims()
		if rV != cV {
			return errors.New("V must be square matrix")
		}
		if rV != rF {
			return errors.New("row dim of V must be matched to row dim of F")
		}

		f.V = mat64.DenseCopyOf(v)
	}

	return nil
}
