# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from grl import ReverseLayerF


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args

        self.domain_classifier = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.Linear(config.hidden_size, 2))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.cycle_criterion = nn.L1Loss()
        #self.tf_criterion = nn.BCEWithLogitsLoss()
        #self.python_tf_classifier = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.Linear(config.hidden_size, 1))
        #self.java_tf_classifier = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.Linear(config.hidden_size, 1))

        #python-java
        self.python_java = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.Linear(config.hidden_size, int(config.hidden_size/2)), nn.ReLU(inplace=True), 
                                nn.Dropout(p=self.args.dropout), nn.Linear(int(config.hidden_size/2), int(config.hidden_size)))  #[dropout-768-384-relu-dropout-768]
                                
        #java-python
        self.java_python = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.Linear(int(config.hidden_size), int(config.hidden_size/2)), nn.ReLU(inplace=True), 
                                nn.Dropout(p=self.args.dropout), nn.Linear(int(config.hidden_size/2), config.hidden_size))  #[dropout-768-384-relu-dropout-768]
        self.sigmod = nn.Sigmoid()

    def forward(self, input_ids=None,p_input_ids=None,n_input_ids=None,labels=None, negative_labels=None, domain_labels=None,alpha=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
        all_labels = torch.cat((labels,labels,negative_labels),0)
        outputss=self.encoder(input_ids,attention_mask=input_ids.ne(1))[1]
        # print('labels', labels)
        # print('negative_labels', negative_labels)
        # print('domain_labels', domain_labels)
        outputs=outputss.split(bs,0)

        prob_1=(outputs[0]*outputs[1]).sum(-1)  #[batch]
        prob_2=(outputs[0]*outputs[2]).sum(-1)   
        temp=torch.cat((outputs[0],outputs[1]),0)  
        temp_labels=torch.cat((labels,labels),0)  
        prob_3= torch.mm(outputs[0],temp.t())  
        mask=labels[:,None]==temp_labels[None,:]   
        prob_3=prob_3*(1-mask.float())-1e9*mask.float()  
        
        prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1)  
        loss=torch.log(prob[:,0]+1e-10)  
        loss=-loss.mean()

        domain_labelss = torch.cat((domain_labels, domain_labels, domain_labels),0)
        # reversed_pooled_output = ReverseLayerF.apply(outputss, alpha)  
        # domain_logits = self.domain_classifier(reversed_pooled_output)
        # domain_loss = self.criterion(domain_logits.contiguous().view(-1, 2).cuda(), domain_labelss.contiguous().view(-1).cuda()).cuda()


        for i in range(bs*3):
            if i==0 and domain_labelss[i]==1:
                forward_vector=self.python_java(outputss[i]).unsqueeze(0)
                cycle_vector=self.java_python(self.python_java(outputss[i])).unsqueeze(0)
            elif i==0 and domain_labelss[i]==0:
                forward_vector=self.java_python(outputss[i]).unsqueeze(0)
                cycle_vector=self.python_java(self.java_python(outputss[i])).unsqueeze(0)
            elif i!=0 and domain_labelss[i]==1:
                forward_vector=torch.cat((forward_vector,self.python_java(outputss[i]).unsqueeze(0)), dim=0)
                cycle_vector=torch.cat((cycle_vector,self.java_python(self.python_java(outputss[i])).unsqueeze(0)), dim=0)
            else:
                forward_vector=torch.cat((forward_vector,self.java_python(outputss[i]).unsqueeze(0)), dim=0)
                cycle_vector=torch.cat((cycle_vector,self.python_java(self.java_python(outputss[i])).unsqueeze(0)), dim=0)

        forward_vector = torch.true_divide(forward_vector, (torch.norm(forward_vector, dim=-1, keepdim=True)+1e-13))  #[batch, hidden_size]
        cycle_vector = torch.true_divide(cycle_vector, (torch.norm(cycle_vector, dim=-1, keepdim=True)+1e-13))  #[batch, hidden_size]
        
        all_domain_labelss = torch.cat((domain_labelss, 1-domain_labelss, domain_labelss),0)
        all_vectors = torch.cat((outputss, forward_vector, cycle_vector),0)
        reversed_pooled_output = ReverseLayerF.apply(all_vectors, alpha) 
        #print('reversed_pooled_output\n',reversed_pooled_output) 
        domain_logits = self.domain_classifier(reversed_pooled_output)
        domain_loss = self.criterion(domain_logits.contiguous().view(-1, 2).cuda(), all_domain_labelss.contiguous().view(-1).cuda()).cuda()

        cycle_loss = self.cycle_criterion(outputss, cycle_vector)
        #print('cycle loss\n', cycle_loss)

        return loss, domain_loss, cycle_loss, outputs[0]




        