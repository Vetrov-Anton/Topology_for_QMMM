import parmed as pmd
from parmed import Atom
import numpy as np


class QM:
    def __init__(self,file_gro, file_top,file_ogro, file_otop, file_ondx):

        self.file_gro = file_gro
        self.file_top = file_top
        
        self.igro = None
        self.itop = None
        self.indx = None 

        self.qm_manual_mask = ''
        self.qm_extend_mask = ''
        self.qm_mask = ''

        self.i_qm_protein = None
        self.o_qm_protein = None
        self.i_rest = None
        self.o_rest = None 

        
        self.qm = None #atoms in QM region

        self.qmmm_bonds = []
        self.mm1_atoms = []
        self.qmqm_bonds = []
        self.vs2 = []
        self.LA_indexes = []
        self.extended = []
        self.breakable_bonds = {'CB':'CA', 'N':'CA', 'CA':'C', 'C':'CA'} #order is important: QM-MM !!!
        self.ions_and_sol = {'SOL','NA','CL'}
        self.H_dist = {('CB','CA') : 1.0 , ('N','CA') : 0.9 , ('CA','C') : 1.0 , ('C','CA') : 1.0} # distances to LA for breakable bonds QM-MM
        self.count = {}
     
        self.aim_qm_charge = None
        self.qm_charge = None
        self.group_to_redistr_charge = ':ALA,PRO,GLY,ILE,VAL,SER,THR,CYS,TRP,ASN,ASP,GLU,GLN,HIS,MET,LEU,TYR,ARG,PHE,LYS'

        self.head = []
        self.tail = []

        self.otop = None
        self.ondx = None
        
        self.file_otop = file_otop
        self.file_ogro = file_ogro
        self.file_ondx = file_ondx
        self.read_inputs()

    def read_head_tail(self):
        with open(self.file_top,'r') as f:
            for line in f:
                if line.strip() == '[ atoms ]':
                    break
                if line[0] != ';':
                    self.head.append(line)
            for line in f:
                if line.strip() == '; Include Position restraint file':
                    break
            for line in f:
                if line[0]!=';':
                    self.tail.append(line)
    def read_inputs(self):
        self.itop = pmd.load_file(self.file_top, xyz = self.file_gro)
        self.read_head_tail()
    
    def choose_qm_manually(self,qm_manual_mask):
        if self.qm_manual_mask == '':
            self.qm_manual_mask = qm_manual_mask
        else:
            self.qm_manual_mask = f'({self.qm_manual_mask})|({qm_manual_mask})'
    
    def dfs_extend(self,start):  
        visited = set()  
        stack = [start]  
        while stack:  
            prev_atom_idx = stack.pop()
            prev_atom = self.itop.atoms[prev_atom_idx]
            if prev_atom_idx not in visited:  
                visited.add(prev_atom_idx)     
            for next_atom in prev_atom.bond_partners:
                if (next_atom.idx not in visited) and (self.breakable_bonds.get(prev_atom.name)!=next_atom.name) and (next_atom.idx != prev_atom.idx):
                    stack.append(next_atom.idx)
        return visited  
    
    def choose_qm_to_extend(self, qm_extend_mask):
        if self.qm_extend_mask == '':
            self.qm_extend_mask = self.extend_until_break(qm_extend_mask)
        else:
            self.qm_extend_mask = f'({self.qm_extend_mask})|({self.extend_until_break(qm_extend_mask)})'

    def extend_until_break(self, amber_mask):
        qm_ind = [atom.idx for atom in self.itop.view[amber_mask].atoms]
        visited = set()
        for index in qm_ind:
            if index not in visited:
                visited = visited.union(self.dfs_extend(index))
        output_mask = ','.join([str(ind+1) for ind in list(visited)])
        output_mask = '@' + output_mask
        return output_mask

    def determine_qm(self):
        if self.qm_manual_mask == '' and self.qm_extend_mask != '':
            self.qm_mask = self.qm_extend_mask
        elif self.qm_manual_mask != '' and self.qm_extend_mask == '':
            self.qm_mask = self.qm_manual_mask
        elif self.qm_manual_mask != '' and self.qm_extend_mask != '':
            self.qm_mask = f'({self.qm_manual_mask})|({self.qm_extend_mask})'
        else:
            pass
        qm_coordinates = set()
        qm = self.itop[self.qm_mask]
        for xyz in qm.coordinates:
            xyz = str(xyz)
            qm_coordinates.add(xyz)
            #print(xyz)
        #print(qm_coordinates)
        #print(self.qm_mask)
        #print(self.qm_extend_mask)
        ions_and_sol_mask = ':' + ','.join(list(self.ions_and_sol))
        self.i_qm_protein = self.itop[f'({self.qm_mask})|(!({ions_and_sol_mask}))']
        #print(self.i_qm_protein.atoms)
        self.o_qm_protein = self.i_qm_protein.copy(type(self.i_qm_protein))
        self.o_rest = self.itop[f'!(({self.qm_mask})|(!({ions_and_sol_mask})))']
        #self.o_rest = self.i_rest.copy(type(self.i_rest))
        new_qm_mask = []
        xyz = self.i_qm_protein.coordinates
        for atom in self.i_qm_protein.atoms:
            if str(xyz[atom.idx]) in qm_coordinates:
                new_qm_mask.append(str(atom.idx + 1))
        self.qm_mask = '@' + ','.join(new_qm_mask)
        self.qm = self.i_qm_protein.view[self.qm_mask]

        for i in self.ions_and_sol:
            self.count[i] = 0
        for res in self.qm.residues:
            if res.name in self.ions_and_sol:
                self.count[res.name] += 1
        #print(self.qm_mask)

    def calculate_charge_qm(self):
        charge = 0
        for atom in self.qm.atoms:
            charge += atom.charge
        self.qm_charge = charge
    
    def redistribute_charge_from_qm_to_mm(self, aim_charge=0):
        self.aim_qm_charge = aim_charge
        qm_idx = []
        mm_idx = []
        charge_redistr = aim_charge - self.qm_charge
        for atom in self.qm.atoms:
            qm_idx.append(atom.idx)
        group = self.i_qm_protein.view[self.group_to_redistr_charge]
        for atom in group.atoms:
            if atom not in self.qm.atoms:
                mm_idx.append(atom.idx)
        dq_qm = charge_redistr/len(qm_idx)
        dq_mm = charge_redistr/len(mm_idx)
        for i in qm_idx:
            self.o_qm_protein.atoms[i].charge += dq_qm
        for i in mm_idx:
            self.o_qm_protein.atoms[i].charge -= dq_mm

    def full_charge(self,system):
        charge = 0
        for atom in system.atoms:
            charge += atom.charge
        return charge
    
    def find_qmqm_bonds(self):
        self.qmqm_bonds = []
        for bond in self.i_qm_protein.bonds:
            a1, a2 = bond.atom1, bond.atom2
            if (a1 in self.qm.atoms) and (a2 in self.qm.atoms):
                self.qmqm_bonds.append(bond)

    def find_qmmm_bonds(self):
        self.qmmm_bonds = []
        self.mm1_atoms = []
        for bond in self.i_qm_protein.bonds:
            a1 , a2 = bond.atom1, bond.atom2
            if a1 in self.qm.atoms and a2 not in self.qm.atoms: # and (a1.name!='SG' or a2.name != 'SG'): #exclude CYS-bonds
                self.qmmm_bonds.append(bond)
                self.mm1_atoms.append(a2.idx)
            elif a1 not in self.qm.atoms and a2 in self.qm.atoms: #and (a1.name!='SG' or a2.name != 'SG'):
                self.qmmm_bonds.append(bond)
                self.mm1_atoms.append(a1.idx)
        self.mm1_atoms = list(set(self.mm1_atoms))

    def process_angles(self): # delete all angles with 2 or 3 QM atoms
        new_angles = []
        for angle in self.o_qm_protein.angles:
            a1, a2, a3 = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            atoms = [a1,a2,a3]
            n_qm_atoms = 0
            for atom in atoms:
                if self.i_qm_protein.atoms[atom] in self.qm.atoms:
                    n_qm_atoms += 1
            if n_qm_atoms in {2,3}:
                continue
            else:
                angle.type = None
                new_angles.append(angle)
        self.o_qm_protein.angles = new_angles
    
    def process_dihedrals(self): # delete all dihedrals with 3 or 4 QM atoms
        new_dihedrals = []
        for dihedral in self.i_qm_protein.dihedrals:
            a1, a2, a3, a4 = dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx
            atoms = [a1,a2,a3,a4]
            n_qm_atoms = 0
            for atom in atoms:
                if self.i_qm_protein.atoms[atom] in self.qm.atoms:
                    n_qm_atoms += 1
            if n_qm_atoms in {3,4}:
                continue
            else:
                dihedral.type = None
                new_dihedrals.append(dihedral)
        self.o_qm_protein.dihedrals = new_dihedrals
    
    def process_impropers(self): # delete all impropers with 3 or 4 QM atoms # might be unnecessary?
        new_impropers = []
        for improper in self.i_qm_protein.impropers:
            a1, a2, a3, a4 = improper.atom1.idx, improper.atom2.idx, improper.atom3.idx, improper.atom4.idx
            atoms = [a1,a2,a3,a4]
            n_qm_atoms = 0
            for atom in atoms:
                if self.i_qm_protein.atoms[atom] in self.qm.atoms:
                    n_qm_atoms += 1
            if n_qm_atoms in {3,4}:
                continue
            else:
                improper.type = None
                new_impropers.append(improper)
        self.o_qm_protein.impropers = new_impropers
    
    def process_bonds(self): # make all QM-QM bonds type 5
        new_bonds = []
        self.find_qmqm_bonds()
        for bond in self.i_qm_protein.bonds:
            if bond in self.qmqm_bonds:
                bond.funct = 5
            bond.type = None
            new_bonds.append(bond)
        self.o_qm_protein.bonds = new_bonds

    def vs2_and_LA(self):
        #self.hsd_qm = self.otop[self.qm_mask]
        self.vs2 = []
        #self.qm_protein = self.otop[f'({self.qm_mask})|(!(:SOL,NA,CL))']
        #self.rest = self.otop[f'!(({self.qm_mask})|(!(:SOL,NA,CL)))']
        #print(len(self.qm_protein.atoms))
        #print(len(self.rest.atoms))
        #print(len(self.otop.atoms))
        #self.rest = self.otop[f'!(({self.qm_mask})|(!(:SOL,NA,CL)))']
        res_n = len(self.i_qm_protein.residues)
        atom_n = len(self.i_qm_protein.atoms)
        for bond in self.qmmm_bonds:
            if bond.atom1 in self.qm.atoms:
                aqm, amm = bond.atom1, bond.atom2
            else:
                aqm, amm = bond.atom2, bond.atom1
            xyz = self.i_qm_protein.coordinates
            qm_xyz = xyz[aqm.idx]
            mm_xyz = xyz[amm.idx]
            LA_xyz = qm_xyz + self.H_dist[(aqm.name,amm.name)] * (mm_xyz-qm_xyz)/(np.linalg.norm(mm_xyz-qm_xyz)) ###
            LA = Atom(name='LA', atomic_number=1, type='LA',charge=0, mass=0)
            LA.xx,LA.xy,LA.xz = LA_xyz[0],LA_xyz[1],LA_xyz[2]
            #LA.list = self.o_qm_protein.atoms
            #self.hsd_qm.add_atom(LA,'XXX',res_n,chain = '')
            #nLA = copy.deepcopy(LA)
            self.o_qm_protein.add_atom(LA,'XXX',res_n,chain = '')
            self.vs2.append([str(atom_n+1), str(aqm.idx+1),str(amm.idx+1), str(1), f'{self.H_dist[(aqm.name,amm.name)]/np.linalg.norm(mm_xyz-qm_xyz):.3f}','; qmmm'])
            #self.LA_indexes.append(atom_n + 1)
            self.qm_mask += f',{atom_n+1}'
            res_n += 1
            atom_n += 1
        #self.qm = self.o_qm_protein.view[self.qm_mask]
    
    def amber_redist(self):
        mm_index = set()
        charge_to_redistr = 0
        for bond in self.qmmm_bonds:
            if bond.atom1 not in self.qm.atoms:
                amm = bond.atom1
            else:
                amm = bond.atom2
            charge_to_redistr += self.o_qm_protein.atoms[amm.idx].charge
            self.o_qm_protein.atoms[amm.idx].charge =  0
            mm_index.add(amm.idx)
        #mm_index = set(mm_index)
        group = self.i_qm_protein.view[self.group_to_redistr_charge]
        n_mm_atoms = 0
        for atom in group.atoms:
            if (atom not in self.qm.atoms) and (atom.idx not in mm_index) and (atom.name != 'LA'):
                n_mm_atoms += 1
        dq = charge_to_redistr/n_mm_atoms
        for atom in group.atoms:
            if (atom not in self.qm.atoms) and (atom.idx not in mm_index) and (atom.name != 'LA'):
                self.o_qm_protein.atoms[atom.idx].charge += dq

    def RC_redist(self):
        res_n = len(self.o_qm_protein.residues)
        atom_n = len(self.o_qm_protein.atoms)
        xyz = self.o_qm_protein.coordinates
        for bond in self.qmmm_bonds:
            if bond.atom1.name == 'CP' or bond.atom2.name == 'CP':
                continue
            if bond.atom1 not in self.qm.atoms:
                mm1 = bond.atom1
            else:
                mm1 = bond.atom2
            dq = self.o_qm_protein.atoms[mm1.idx].charge/(len(mm1.bond_partners) - 1)
            self.o_qm_protein.atoms[mm1.idx].charge = 0
            for mm2 in mm1.bond_partners:
                if mm2 not in self.qm.atoms and mm2.name != 'CP':
                    CP_xyz = xyz[mm1.idx] + (xyz[mm2.idx]-xyz[mm1.idx])/2
                    CP = Atom(name='CP', atomic_number=1, type='CP',charge=dq, mass=0,)
                    CP.xx,CP.xy,CP.xz = CP_xyz[0],CP_xyz[1],CP_xyz[2]
                    self.o_qm_protein.add_atom(CP,'XXX',res_n,chain = '')
                    self.vs2.append([str(atom_n+1), str(mm1.idx+1),str(mm2.idx+1), str(1), f'{0.500:.3f}','; RC'])
                    bond5 = pmd.Bond(mm1,CP)
                    bond5.funct = 5
                    self.o_qm_protein.bonds.append(bond5)
                    res_n += 1
                    atom_n += 1
                    
    def RCD_redist(self):
        res_n = len(self.o_qm_protein.residues)
        atom_n = len(self.o_qm_protein.atoms)
        xyz = self.o_qm_protein.coordinates
        for bond in self.qmmm_bonds:
            if bond.atom1.name == 'CP' or bond.atom2.name == 'CP':
                continue
            if bond.atom1 not in self.qm.atoms:
                mm1 = bond.atom1
            else:
                mm1 = bond.atom2
            dq = self.o_qm_protein.atoms[mm1.idx].charge/(len(mm1.bond_partners) - 1)
            self.o_qm_protein.atoms[mm1.idx].charge = 0
            for mm2 in mm1.bond_partners:
                if mm2 not in self.qm and mm2.name != 'CP':
                    CP_xyz = xyz[mm1.idx] + (xyz[mm2.idx]-xyz[mm1.idx])/2
                    CP = Atom(name='CP', atomic_number=1, type='CP',charge=2*dq, mass=0,)
                    CP.xx,CP.xy,CP.xz = CP_xyz[0],CP_xyz[1],CP_xyz[2]
                    self.o_qm_protein.add_atom(CP,'XXX',res_n,chain = '')
                    self.o_qm_protein.atoms[mm2.idx].charge -= dq
                    self.vs2.append([str(atom_n+1), str(mm1.idx+1),str(mm2.idx+1), str(1), f'{0.500:.3f}','; RCD'])
                    bond5 = pmd.Bond(mm1,CP)
                    bond5.funct = 5
                    self.o_qm_protein.bonds.append(bond5)
                    res_n += 1
                    atom_n += 1

    def CS_redist(self):
        res_n = len(self.o_qm_protein.residues)
        atom_n = len(self.o_qm_protein.atoms)
        xyz = self.o_qm_protein.coordinates
        for bond in self.qmmm_bonds:
            if bond.atom1.name == 'CP' or bond.atom2.name == 'CP':
                continue
            if bond.atom1 not in self.qm.atoms:
                mm1 = bond.atom1
            else:
                mm1 = bond.atom2
            dq = self.o_qm_protein.atoms[mm1.idx].charge/(len(mm1.bond_partners) - 1)
            self.o_qm_protein.atoms[mm1.idx].charge = 0
            for mm2 in mm1.bond_partners:
                if mm2 not in self.qm and mm2.name != 'CP':
                    CP1_xyz = xyz[mm1.idx] + (xyz[mm2.idx]-xyz[mm1.idx])*0.940
                    CP1 = Atom(name='CP', atomic_number=1, type='CP',charge=dq, mass=0,)
                    CP1.xx,CP1.xy,CP1.xz = CP1_xyz[0],CP1_xyz[1],CP1_xyz[2]
                    self.o_qm_protein.add_atom(CP1,'XXX',res_n,chain = '')
                    self.o_qm_protein.atoms[mm2.idx].charge += dq
                    self.vs2.append([str(atom_n+1), str(mm1.idx+1),str(mm2.idx+1), str(1), f'{0.940:.3f}','; charge shift'])
                    bond5 = pmd.Bond(mm1,CP1)
                    bond5.funct = 5
                    self.o_qm_protein.bonds.append(bond5)
                    res_n += 1
                    atom_n += 1

                    CP2_xyz = xyz[mm1.idx] + (xyz[mm2.idx]-xyz[mm1.idx])*1.060
                    CP2 = Atom(name='CP', atomic_number=1, type='CP',charge=-dq, mass=0,)
                    CP2.xx,CP2.xy,CP2.xz = CP2_xyz[0],CP2_xyz[1],CP2_xyz[2]
                    self.o_qm_protein.add_atom(CP2,'XXX',res_n,chain = '')
                    self.vs2.append([str(atom_n+1), str(mm1.idx+1),str(mm2.idx+1), str(1), f'{1.060:.3f}','; charge shift'])
                    bond5 = pmd.Bond(mm1,CP2)
                    bond5.funct = 5
                    self.o_qm_protein.bonds.append(bond5)
                    res_n += 1
                    atom_n += 1
    
    def write_ndx(self):
        self.qm = self.o_qm_protein.view[self.qm_mask]
        #print(self.qm.atoms,'rty')
        with open(self.file_ondx,'w') as f:
            #for i in self.indx:
                #print(i, file=f)
            print('[ QM ]', file=f)
            count = 0
            for atom in self.qm.atoms:
                if count > 20:
                    print('\n',end='',file=f)
                    count = 0
                print(atom.idx + 1,end = ' ', file=f) #start index 1
                count += 1
            #for LA in self.LA_indexes:
                #print(LA,end = ' ', file=f)
            print('',file=f)
            print('[ freeze ]', file=f)
            count = 0
            for atom in self.o_qm_protein.atoms:
                if count > 20:
                    print('\n',end='',file=f)
                    count = 0
                print(atom.idx + 1,end = ' ', file=f) #start index 1
                count += 1
            print('',file=f)
            print('[ Water_and_ions ]', file=f)
            count = 0
            for atom in self.o_rest.atoms:
                if count > 20:
                    print('\n',end='',file=f)
                    count = 0
                print(atom.idx + len(self.o_qm_protein.atoms)+1,end = ' ', file=f)
                count += 1 #start index 1
            print('',file=f)\
            
    def rewrite(self):
        main = []
        flag = False
        with open(self.file_otop,'r') as f:
            for line in f:
                if line.strip() == '[ atoms ]':
                    flag = True
                if line.strip() == '[ system ]':
                    break
                if flag:
                    main.append(line)
        with open(self.file_otop,'w') as f:
            for line in self.head:
                print(line,file=f,end='')
            for line in main:
                print(line,file=f,end='')
            print(' [ cmap ]',file=f)
            print(';  ai    aj    ak    al    am funct',file=f)
            print('',file=f)
            print(' [ virtual_sites2 ]',file=f)
            print(';  ai    aj    ak    funct            c0',file=f)
            col_widths = [max(len(row[i]) for row in self.vs2) for i in range(len(self.vs2[0]))]
            for row in self.vs2:
                print('    '+'     '.join(f"{item:>{col_widths[i]}}" for i, item in enumerate(row)),file=f)
            for line in self.tail:
                flag = True
                for i in list(self.ions_and_sol):
                    if line.startswith(i):
                        name, n = line.split()
                        n = int(n)
                        print(f'{name}      {n-self.count[name]}',file=f)
                        flag = False
                if flag:
                    print(line,file=f,end='')
    
    def write_otop(self):
        self.o_qm_protein.save(self.file_otop,overwrite=True,format='gromacs',combine='all')
    def write_ogro(self):
        bonds = []
        for bond in self.o_qm_protein.bonds:
            if bond.atom1.name != 'CP' and bond.atom2.name != 'CP':
                bonds.append(bond)
        self.o_qm_protein.bonds = bonds
        top = self.o_qm_protein + self.o_rest
        top.save(self.file_ogro, overwrite=True)
    def write_outputs(self):
        self.write_ndx()
        self.write_otop()
        self.write_ogro()
        self.rewrite()
    
    
    def make_hsd(self,file_hsd):
        #print(self.qm_mask,'djn')
        self.qm = self.o_qm_protein.view[self.qm_mask]
        type2element = {
        "c3":"C","o":"O", "no":"N", 
        "Br": "Br",
        "C": "C", "CA": "C", "CB": "C", "CC": "C", "CK": "C", "CM": "C", "CN": "C", "CQ": "C", "CR": "C", "CT": "C", "CV": "C", "CW": "C", "C*": "C", "C0": "C",'CG':'C','CD1':'C',
        "F": "F", "f": "F",
        "H": "H", "HC": "H", "H1": "H", "H2": "H", "H3": "H", "HA": "H", "H4": "H", "H5": "H", "HO": "H", "HS": "H", "HW": "H","HP": "H",'HB1':'H','HB2':'H', 'HB3':'H',
        "I": "I",
        "fCa": "Ca",
        "Cl": "Cl",
        "Na": "Na",
        "MG": "Mg",
        "N": "N", "NA": "N", "NB": "N", "NC": "N", "N2": "N", "N3": "N", "N*": "N",
        "O": "O", "OW": "O", "OH": "O", "OS": "O", "O2": "O",
        "P": "P", "p5": "P",
        "S": "S", "ss": "S",
        "SH": "S",
        "CU": "Cu",
        "FE": "Fe",
        "K": "K",
        "Rb": "Rb",
        "Cs": "Cs",
        "OW_spc": "O", "OW_tip4pew": "O", "OW_tip4p": "O", "OW_tip5p": "O", 'OW_tip3pfb':'O',
        "HW_spc": "H", "HW_tip4pew": "H", "HW_tip4p": "H", "HW_tip5p": "H", 'HW_tip3pfb':'H',
        "Li": "Li",
        "Zn": "Zn",
        "LA": "H",
        "ho": "H", "hn": "H", "hc": "H", "hx": "H", "h1": "H", "h2": "H", "ha": "H",
        "o": "O", "oh": "O", "os": "O",
        "c": "C", "c3": "C", "ca": "C",
        "n": "N", "n4": "N", "gp5": "P", "gos": "O", "go": "O", "gc3": "C", "gh1": "H", "gca": "C", "gha": "H", "gno": "N", "ghc": "H",
        "zl": "P", "zh": "O", "zi": "O", "zj": "O", "za": "C", "zb": "C", "zc": "C",
        "zd": "C", "ze" : "C", "zf": "C", "zg": "N", "zk": "O",
        "X1": "C", "X2": "C", "X3":"C", "X4":"C", "X5":"C", "X6":"C", "X7":"C","X8":"C","X9":"C","X10":"C","X11":"C","X12":"C","X13":"C","X14":"C","X15":"C","X16":"C",
        "2C": "C", "CO": "C", "3C": "C", "dza": "C", "dzb": "C", "dze": "O", "dzd": "O", "dzf": "P", "dzc": "F", "dh1": "H", "dhc": "H"
        }
        MaxAngularMomentum = {
            'C': 'p', 'O': 'p', 'N': 'p', 'H': 's', 'P': 'd', 'S': 'd',
            'Br': 'd', 'Cl': 'd', 'F': 'p', 'Ca': 'p',
            'Zn': 'd', 'Mg': 'p', 'Na': 'p', 'K': 'p', 'Fe': 'd', 'Cu': 'd', 'Li': 'd'
        }
        HubbardDerivs = {
            'Br': -0.0573,
            'C': -0.1492,
            'Ca': -0.034,
            'Cl': -0.0697,
            'F': -0.1623,
            'H': -0.1857,
            'I': -0.0433,
            'K': -0.0339,
            'Mg': -0.02,
            'N': -0.1535,
            'Na': -0.0454,
            'O': -0.1575,
            'P': -0.14,
            'S': -0.11,
            'Zn': -0.03,
        }
        skpath = "/home/domain/data/zlobin/dftb-par/3ob-3-1-ophyd/"

        elements = list(set([type2element[atom.type] for atom in self.qm.atoms]))
        #print(elements)
        txt = ''
        elements_str = ' '.join(elements)
        txt = f"Geometry =  {{ \n   TypeNames = {elements_str}\n  TypesAndCoordinates {{\n"
        xyz = self.o_qm_protein.coordinates
        for atom in self.qm.atoms:
            a_xyz = xyz[atom.idx]
            element_index = elements.index(type2element[atom.type]) + 1
            txt += f"{element_index}{a_xyz[0]:8.3f}{a_xyz[1]:8.3f}{a_xyz[2]:8.3f}\n"

        txt += "}\n}\n"

        txt += f"    Hamiltonian = DFTB {{\n"
        txt += f"    SCC = Yes\n"
        txt += f"    SCCTolerance = 1e-6\n"
        txt += f"    Charge = {int(self.aim_qm_charge)}\n"
        txt += f"    MaxAngularMomentum {{\n"

        for e in elements:
            txt += f'    {e} = "{MaxAngularMomentum[e]}"\n'

        txt += f"    }}\n"
        txt += f"    Dispersion = DftD4 {{\n"
        txt += f"      s10 = 0\n"
        txt += f"      s6 = 1\n"
        txt += f"      s8 = 0.4727337\n"
        txt += f"      s9 = 0\n"
        txt += f"      a1 = 0.5467502\n"
        txt += f"      a2 = 4.4955068\n"
        txt += f"    }}\n"
        txt += f"    SlaterKosterFiles = Type2FileNames {{\n"
        txt += f"    Prefix = {skpath}\n"
        txt += f"    Separator = \"\"\n"
        txt += f"    LowerCaseTypeName = Yes\n"
        txt += f"    Suffix = \"-c.spl\"   }}\n"
        txt += f"    ThirdOrderFull = Yes\n"
        txt += f"    HubbardDerivs {{\n"

        for e in elements:
            txt += f'    {e} = {HubbardDerivs[e]}\n'

        txt += f"    }}\n"
        txt += f"      HCorrection = Damping {{\n"
        txt += f"        Exponent = 4.05\n"
        txt += f"      }}\n"
        txt += f"   }}\n"
        txt += f"    Analysis = {{\n"
        txt += f"      CalculateForces = Yes\n"
        txt += f"      ProjectStates = {{}}\n"
        txt += f"      WriteEigenvectors = No\n"
        txt += f"      WriteBandOut = No\n"
        txt += f"      MullikenAnalysis = No\n"
        txt += f"      AtomResolvedEnergies = No\n"
        txt += f"    }}\n"
        txt += f"    Options = {{\n"
        txt += f"      WriteDetailedOut = No\n"
        txt += f"      WriteAutotestTag = No\n"
        txt += f"      WriteDetailedXML = No\n"
        txt += f"      WriteResultsTag = No\n"
        txt += f"      RestartFrequency = 2000\n"
        txt += f"      RandomSeed = 0\n"
        txt += f"      WriteHS = No\n"
        txt += f"      WriteRealHS = No\n"
        txt += f"      MinimiseMemoryUsage = No\n"
        txt += f"      ShowFoldedCoords = No\n"
        txt += f"      TimingVerbosity = 0\n"
        txt += f"      WriteChargesAsText = No\n"
        txt += f"    }}\n"
        with open(file_hsd, 'w') as hsd:
            hsd.write(txt)

    def job(self,qm_aim_charge = 0, scheme = 'amber'):
        self.determine_qm()
        self.calculate_charge_qm()
        self.redistribute_charge_from_qm_to_mm(aim_charge=qm_aim_charge)
        self.find_qmqm_bonds()
        self.find_qmmm_bonds()
        self.process_angles()
        self.process_dihedrals()
        self.process_impropers()
        self.process_bonds()
        self.vs2_and_LA()
        if scheme == 'amber':
            self.amber_redist() 
        elif scheme == 'RC':
            self.RC_redist()
        elif scheme == 'RCD':
            self.RCD_redist()
        elif scheme == 'CS':
            self.CS_redist()
        self.write_outputs()


def rewrite_hsd(xyz_file,qm_charge,top_file='qm.top', ndx_file='qm.ndx', o_hsd='dftb_in.hsd', o_gro='qm_new.gro'): #Change coordinates in hsd from xyz_file(pdb, gro)
    qm_idx = []
    with open(ndx_file,'r') as f:
        for line in f:
            if line.startswith('[ QM ]'):
                break
        for line in f:
            if line.startswith('['):
                break
            idx = list(map(str, line.strip().split()))
            qm_idx += idx
    amber_mask = '@' + ','.join(qm_idx)
    top = pmd.load_file(top_file, xyz = xyz_file)
    qm = top.view[amber_mask]
    print(qm.atoms)
    type2element = {
        "c3":"C","o":"O", "no":"N", 
        "Br": "Br",
        "C": "C", "CA": "C", "CB": "C", "CC": "C", "CK": "C", "CM": "C", "CN": "C", "CQ": "C", "CR": "C", "CT": "C", "CV": "C", "CW": "C", "C*": "C", "C0": "C",'CG':'C','CD1':'C',
        "F": "F", "f": "F",
        "H": "H", "HC": "H", "H1": "H", "H2": "H", "H3": "H", "HA": "H", "H4": "H", "H5": "H", "HO": "H", "HS": "H", "HW": "H","HP": "H",'HB1':'H','HB2':'H', 'HB3':'H',
        "I": "I",
        "fCa": "Ca",
        "Cl": "Cl",
        "Na": "Na",
        "MG": "Mg",
        "N": "N", "NA": "N", "NB": "N", "NC": "N", "N2": "N", "N3": "N", "N*": "N",
        "O": "O", "OW": "O", "OH": "O", "OS": "O", "O2": "O",
        "P": "P", "p5": "P",
        "S": "S", "ss": "S",
        "SH": "S",
        "CU": "Cu",
        "FE": "Fe",
        "K": "K",
        "Rb": "Rb",
        "Cs": "Cs",
        "OW_spc": "O", "OW_tip4pew": "O", "OW_tip4p": "O", "OW_tip5p": "O", 'OW_tip3pfb':'O',
        "HW_spc": "H", "HW_tip4pew": "H", "HW_tip4p": "H", "HW_tip5p": "H", 'HW_tip3pfb':'H',
        "Li": "Li",
        "Zn": "Zn",
        "LA": "H",
        "ho": "H", "hn": "H", "hc": "H", "hx": "H", "h1": "H", "h2": "H", "ha": "H",
        "o": "O", "oh": "O", "os": "O",
        "c": "C", "c3": "C", "ca": "C",
        "n": "N", "n4": "N", "gp5": "P", "gos": "O", "go": "O", "gc3": "C", "gh1": "H", "gca": "C", "gha": "H", "gno": "N", "ghc": "H",
        "zl": "P", "zh": "O", "zi": "O", "zj": "O", "za": "C", "zb": "C", "zc": "C",
        "zd": "C", "ze" : "C", "zf": "C", "zg": "N", "zk": "O",
        "X1": "C", "X2": "C", "X3":"C", "X4":"C", "X5":"C", "X6":"C", "X7":"C","X8":"C","X9":"C","X10":"C","X11":"C","X12":"C","X13":"C","X14":"C","X15":"C","X16":"C",
        "2C": "C", "CO": "C", "3C": "C", "dza": "C", "dzb": "C", "dze": "O", "dzd": "O", "dzf": "P", "dzc": "F", "dh1": "H", "dhc": "H"
        }
    MaxAngularMomentum = {
        'C': 'p', 'O': 'p', 'N': 'p', 'H': 's', 'P': 'd', 'S': 'd',
        'Br': 'd', 'Cl': 'd', 'F': 'p', 'Ca': 'p',
        'Zn': 'd', 'Mg': 'p', 'Na': 'p', 'K': 'p', 'Fe': 'd', 'Cu': 'd', 'Li': 'd'
    }
    HubbardDerivs = {
        'Br': -0.0573,
        'C': -0.1492,
        'Ca': -0.034,
        'Cl': -0.0697,
        'F': -0.1623,
        'H': -0.1857,
        'I': -0.0433,
        'K': -0.0339,
        'Mg': -0.02,
        'N': -0.1535,
        'Na': -0.0454,
        'O': -0.1575,
        'P': -0.14,
        'S': -0.11,
        'Zn': -0.03,
    }
    skpath = "/home/domain/data/zlobin/dftb-par/3ob-3-1-ophyd/"

    elements = list(set([type2element[atom.type] for atom in qm.atoms]))
    #print(elements)
    txt = ''
    elements_str = ' '.join(elements)
    txt = f"Geometry =  {{ \n   TypeNames = {elements_str}\n  TypesAndCoordinates {{\n"
    xyz = top.coordinates
    i = 1
    for atom in qm.atoms:
        if i in [91,160,92,161,108,109,107]:
            print(i,atom.name,atom.idx+1)
        a_xyz = xyz[atom.idx]
        element_index = elements.index(type2element[atom.type]) + 1
        txt += f"{element_index}{a_xyz[0]:8.3f}{a_xyz[1]:8.3f}{a_xyz[2]:8.3f}\n"
        i+= 1

    txt += "}\n}\n"

    txt += f"    Hamiltonian = DFTB {{\n"
    txt += f"    SCC = Yes\n"
    txt += f"    SCCTolerance = 1e-6\n"
    txt += f"    Charge = {int(qm_charge)}\n"
    txt += f"    MaxAngularMomentum {{\n"

    for e in elements:
        txt += f'    {e} = "{MaxAngularMomentum[e]}"\n'

    txt += f"    }}\n"
    txt += f"    Dispersion = DftD4 {{\n"
    txt += f"      s10 = 0\n"
    txt += f"      s6 = 1\n"
    txt += f"      s8 = 0.4727337\n"
    txt += f"      s9 = 0\n"
    txt += f"      a1 = 0.5467502\n"
    txt += f"      a2 = 4.4955068\n"
    txt += f"    }}\n"
    txt += f"    SlaterKosterFiles = Type2FileNames {{\n"
    txt += f"    Prefix = {skpath}\n"
    txt += f"    Separator = \"\"\n"
    txt += f"    LowerCaseTypeName = Yes\n"
    txt += f"    Suffix = \"-c.spl\"   }}\n"
    txt += f"    ThirdOrderFull = Yes\n"
    txt += f"    HubbardDerivs {{\n"

    for e in elements:
        txt += f'    {e} = {HubbardDerivs[e]}\n'

    txt += f"    }}\n"
    txt += f"      HCorrection = Damping {{\n"
    txt += f"        Exponent = 4.05\n"
    txt += f"      }}\n"
    txt += f"   }}\n"
    txt += f"    Analysis = {{\n"
    txt += f"      CalculateForces = Yes\n"
    txt += f"      ProjectStates = {{}}\n"
    txt += f"      WriteEigenvectors = No\n"
    txt += f"      WriteBandOut = No\n"
    txt += f"      MullikenAnalysis = No\n"
    txt += f"      AtomResolvedEnergies = No\n"
    txt += f"    }}\n"
    txt += f"    Options = {{\n"
    txt += f"      WriteDetailedOut = No\n"
    txt += f"      WriteAutotestTag = No\n"
    txt += f"      WriteDetailedXML = No\n"
    txt += f"      WriteResultsTag = No\n"
    txt += f"      RestartFrequency = 2000\n"
    txt += f"      RandomSeed = 0\n"
    txt += f"      WriteHS = No\n"
    txt += f"      WriteRealHS = No\n"
    txt += f"      MinimiseMemoryUsage = No\n"
    txt += f"      ShowFoldedCoords = No\n"
    txt += f"      TimingVerbosity = 0\n"
    txt += f"      WriteChargesAsText = No\n"
    txt += f"    }}\n"
    with open(o_hsd, 'w') as hsd:
        hsd.write(txt)
    top.save(o_gro,overwrite = True,combine='all')
    



    




    

    



