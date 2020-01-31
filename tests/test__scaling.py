

def peaklist_from_masses_and_rt_grid(masses, rt_grid):
    peaklist = pd.DataFrame(index=rt_cuts, columns=masses).unstack().reset_index()
    del peaklist[0]
    peaklist.columns = ['peakMz', 'rtmin']
    peaklist['rtmax'] = peaklist.rtmin+(1*dt)
    peaklist['peakLabel'] =  peaklist.peakMz.apply(lambda x: '{:.3f}'.format(x)) + '__' + peaklist.rtmin.apply(lambda x: '{:2.2f}'.format(x))
    peaklist['peakMzWidth[ppm]'] = 10
    return peaklist

class TestClass():

    peaklist = get_mass_rt_grid(range(100, 1000, 100), rt_grid=range(1, 11))
    mint = Mint()
    mint.peaklist = peaklist
    print(mint.peaklist)
    print(len(mint.peaklist))
    mint.files = ['tests/data/test.mzXML']*10
    
    def run():
        mint.files = ['tests/data/test.mzXML']*10
        mint.run()
