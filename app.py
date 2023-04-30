import pickle
import os

import sklearn
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from os import path
import hashlib
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

model = pickle.load(open('RFModel.pkl', 'rb'))

app = Flask(__name__)



@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        valid = set()
        valid_list=['aaa', 'aad', 'aam', 'aas', 'adc', 'adcb', 'adcl', 'add', 'addb', 'addl', 'addpd', 'addr16', 'addsd' , 'addw', 'and', 'andb', 'andl', 'andpd', 'andps', 'andw', 'arpl', 'bnd', 'bound', 'bsf', 'bsr', 'bswap', 'bt', 'btl', 'btr', 'bts', 'btsl', 'btw', 'call', 'callq', 'cbtw', 'clc', 'cld', 'cli', 'cltd', 'cltq', 'clts', 'cmc', 'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmove', 'cmovg', 'cmovge', 'cmovl', 'cmovne', 'cmovno', 'cmovns', 'cmovo', 'cmovs', 'cmp', 'cmpb', 'cmpeqsd', 'cmpl', 'cmpnlepd', 'cmpps', 'cmpq', 'cmpsb', 'cmpsl', 'cmpw', 'cpuid', 'cs', 'cvtdq2pd', 'cvtps2pd', 'cvtps2pi', 'cvtsd2si', 'cvtsi2sd', 'cvttsd2si', 'cwtl', 'daa', 'das', 'data16', 'dec', 'decb', 'decl', 'decq', 'decw', 'div', 'divb', 'divl', 'divsd', 'ds', 'enter', 'es', 'f2xm1', 'fabs', 'fadd', 'faddl', 'faddp', 'fadds', 'fbstp', 'fchs', 'fclex', 'fcmovbe', 'fcmovnb', 'fcmovnbe', 'fcmovne', 'fcmovu', 'fcom', 'fcoml', 'fcomp', 'fcompl', 'fcompp', 'fcomps', 'fcoms', 'fcos', 'fdiv', 'fdivl', 'fdivp', 'fdivr', 'fdivrl', 'fdivrp', 'fdivrs', 'fdivs', 'ffree', 'fiaddl', 'fiadds', 'ficoml', 'ficompl', 'ficomps', 'ficoms', 'fidivl', 'fidivrl', 'fidivrs', 'fidivs', 'fildl', 'fildll', 'filds', 'fimull', 'fimuls', 'fistl', 'fistpl', 'fistpll', 'fistps', 'fisttpl', 'fisttpll', 'fisttps', 'fisubrl', 'fisubrs', 'fisubs', 'fld', 'fld1', 'fldcw', 'fldenv', 'fldl', 'fldl2e', 'fldlg2', 'fldln2', 'fldpi', 'flds', 'fldt', 'fldz', 'fmul', 'fmull', 'fmulp', 'fmuls', 'fnclex', 'fninit', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'fprem', 'frndint', 'frstor', 'fs', 'fsave', 'fscale', 'fst', 'fstcw', 'fstl', 'fstp', 'fstpl', 'fstps', 'fstpt', 'fsts', 'fstsw', 'fsub', 'fsubl', 'fsubp', 'fsubr', 'fsubrl', 'fsubrp', 'fsubrs', 'fsubs', 'ftst', 'fucom', 'fucomip', 'fucomp', 'fucompp', 'fwait', 'fxam', 'fxch', 'fxsave', 'fyl2x', 'getsec', 'gs', 'hlt', 'icebp', 'idiv', 'idivb', 'idivl', 'imul', 'imull', 'imulw', 'in', 'inc', 'incb', 'incl', 'incq', 'incw', 'insb', 'insl', 'insw', 'int', 'int3', 'into', 'iret', 'ja', 'jae', 'jb', 'jbe', 'je', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jmpq', 'jne', 'jno', 'jnp', 'jns', 'jo', 'jp', 'jrcxz', 'js', 'lahf', 'lcall', 'ldmxcsr', 'lds', 'lea', 'leave', 'les', 'ljmp', 'lldt', 'lock', 'lods', 'loop', 'loope', 'loopne', 'lret', 'ltr', 'mov', 'movabs', 'movapd', 'movaps', 'movb', 'movd', 'movdqa', 'movdqu', 'movl', 'movlhps', 'movlpd', 'movnti', 'movq', 'movsb', 'movsbl', 'movsbq', 'movsbw', 'movsd', 'movsl', 'movslq', 'movsw', 'movswl', 'movswq', 'movups', 'movw', 'movzbl', 'movzwl', 'mul', 'mull', 'mulpd', 'mulsd', 'neg', 'negb', 'negl', 'nop', 'nopl', 'not', 'notb', 'notl', 'or', 'orb', 'orl', 'orpd', 'orw', 'out', 'outsb', 'outsl', 'outsw', 'paddsw', 'palignr', 'pand', 'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtd', 'pcmpistri', 'pextrw', 'pinsrb', 'pinsrw', 'pmaxsw', 'pminub', 'pmovmskb', 'pmulhw', 'pop', 'popa', 'popaw', 'popf', 'popl', 'por', 'prefetchnta', 'prefetchw', 'psadbw', 'pshufb', 'pshufd', 'pshuflw', 'pshufw', 'psllq', 'psrldq', 'psrlq', 'psubd', 'psubq', 'push', 'pusha', 'pushf', 'pushl', 'pxor', 'rcl', 'rclb', 'rcll', 'rcr', 'rcrb', 'rcrl', 'rep', 'repnz', 'repz', 'ret', 'retq', 'rex', 'rol', 'rolb', 'roll', 'ror', 'rorb', 'rorl', 'sahf', 'sar', 'sarb', 'sarl', 'sbb', 'sbbb', 'sbbl', 'scas', 'seta', 'setae', 'setb', 'sete', 'setg', 'setge', 'setl', 'setle', 'setne', 'setns', 'seto', 'sets', 'sgdtl', 'shl', 'shlb', 'shld', 'shll', 'shr', 'shrb', 'shrd', 'shrl', 'shufps', 'sldt', 'ss', 'stc', 'std', 'sti', 'stmxcsr', 'stos', 'str', 'sub', 'subb', 'subl', 'subpd', 'subsd', 'syscall', 'sysret', 'test', 'testb', 'testl', 'testw', 'ucomisd', 'ud0', 'ud1', 'unpckhpd', 'unpckhps', 'unpcklpd', 'verw', 'vpcmpeqb', 'vpcmpeqw', 'vpmovmskb', 'vpxor', 'vxorps', 'vzeroupper', 'wrmsr', 'xchg', 'xgetbv', 'xlat', 'xor', 'xorb', 'xorl', 'xorpd', 'xorps']

        for x in valid_list:
            valid.add(x)
        f.save(f.filename)
        os.system(f'objdump -d {f.filename} > out.txt')
        file = open(r"out.txt", "rt")
        data = file.read()
        words = data.split()
        # data1 = pd.read_csv(f.filename)
        # data1.drop(data1.columns[data1.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        # opcode1 = (data1["Opcode"].dropna().tolist())
        opcode = [i for i in words if i in valid]
        opc = " ".join(opcode)
        corpus=[]
        corpus.append(opc)
        corpus.append(" ".join(valid_list))
        # opcode = [i for i in opcode1 if i in valid]
        # opc = " ".join(opcode)
        # corpus = []
        # corpus.append(opc)
        # corpus.append(" ".join(valid_list))
        cv1 = CountVectorizer()
        X1 = cv1.fit_transform(corpus).toarray()
        v1 = np.array(X1).astype(np.float32)
        for i in range(len(corpus)):
            s = sum(X1[i])
            # print(s)
            if s == 0:
                v1[i] = 0
            else:
                v1[i] = ((X1[i] / s) * 100).astype(np.float32)
        y11_pred = []
        y11_pred = model.predict(v1)
        # return {
        #     "0": str(y11_pred[0]),
        #     "1": str(y11_pred[1])
        # }
        if y11_pred[0] == 0:
            return "Not Malware"
        else:
            return "Malware"

    return 'Failed'


if __name__ == '__main__':
    app.run(debug=True)