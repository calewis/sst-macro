/**
Copyright 2009-2017 National Technology and Engineering Solutions of Sandia, 
LLC (NTESS).  Under the terms of Contract DE-NA-0003525, the U.S.  Government 
retains certain rights in this software.

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly 
owned subsidiary of Honeywell International, Inc., for the U.S. Department of 
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Copyright (c) 2009-2017, NTESS

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of Sandia Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Questions? Contact sst-macro-help@sandia.gov
*/
#include "computeVisitor.h"
#include "recurseAll.h"
#include "validateScope.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

void
ComputeVisitor::visitAccessDeclRefExpr(DeclRefExpr* expr, MemoryLocation& mloc)
{
  NamedDecl* decl = expr->getFoundDecl();
  Variable& var = variables[decl];
  mloc.append(var.id);
  mloc.updateGeneration(var.generation);
  //std::cout << "access depends on variable " << (char) var.id
  //          << " at generation " << var.generation
  //          << " -> " << mloc.maxGen << std::endl;
}

void
ComputeVisitor::visitAccessCompoundStmt(CompoundStmt* stmt, Loop::Body& body, MemoryLocation& mloc)
{
  errorAbort(stmt->getLocStart(), CI, "visiting compound statement in data flow analysis");
}

void
ComputeVisitor::visitAccessBinaryOperator(BinaryOperator* op, Loop::Body& body, MemoryLocation& mloc)
{
  checkMemoryAccess(op->getLHS(), body, mloc);
  mloc.append(op->getOpcode());
  checkMemoryAccess(op->getRHS(), body, mloc);
}

void
ComputeVisitor::visitAccessParenExpr(ParenExpr* expr, Loop::Body& body, MemoryLocation& mloc)
{
  Expr* subExpr = expr->getSubExpr();
  bool needParens = true;
  switch(subExpr->getStmtClass()){
    case Stmt::ParenExprClass:
    case Stmt::DeclRefExprClass:
    case Stmt::ArraySubscriptExprClass:
      needParens = false;
      break;
    default:
      break;
  }

  if (needParens) mloc.append('(');
  checkMemoryAccess(subExpr, body, mloc);
  if (needParens) mloc.append(')');
}

void
ComputeVisitor::visitAccessImplicitCastExpr(ImplicitCastExpr* expr, Loop::Body& body, MemoryLocation& mloc)
{
  checkMemoryAccess(expr->getSubExpr(), body, mloc);
}

void
ComputeVisitor::visitAccessCStyleCastExpr(CStyleCastExpr* expr, Loop::Body& body, MemoryLocation& mloc)
{
  checkMemoryAccess(expr->getSubExpr(), body, mloc);
}

void
ComputeVisitor::visitAccessArraySubscriptExpr(
  ArraySubscriptExpr* expr, 
  Loop::Body& body, MemoryLocation& mloc,
  bool updateDependence)
{
  MemoryLocation subLoc;
  checkMemoryAccess(expr->getBase(),body,subLoc); //not a lhs
  subLoc.append('[');
  checkMemoryAccess(expr->getIdx(),body,subLoc); //not a lhs
  subLoc.append(']');
  AccessHistory& access = arrays[subLoc];
  if (updateDependence){
    mloc.updateGeneration(access.lastWriteGeneration);
    //std::cout << "access depends on " << subLoc.c_str() << " at generation "
    //          << access.lastWriteGeneration << "->" << mloc.maxGen << std::endl;
  }
  mloc.append(subLoc);
}

void
ComputeVisitor::checkMemoryAccess(clang::Stmt* stmt, Loop::Body& body, MemoryLocation& mloc)
{
#define access_case(type,stmt,...) \
  case(clang::Stmt::type##Class): \
    visitAccess##type(clang::cast<type>(stmt),__VA_ARGS__); break
  switch(stmt->getStmtClass()){
    access_case(DeclRefExpr,stmt,mloc);
    access_case(BinaryOperator,stmt,body,mloc);
    access_case(ParenExpr,stmt,body,mloc);
    access_case(ArraySubscriptExpr,stmt,body,mloc);
    access_case(ImplicitCastExpr,stmt,body,mloc);
    access_case(CStyleCastExpr,stmt,body,mloc);
    access_case(CompoundStmt,stmt,body,mloc);
    default:
      break;
  }
#undef access_case
}

void
ComputeVisitor::visitLoop(ForStmt* stmt, Loop& loop)
{
  ForLoopSpec spec;
  getInitialVariables(stmt->getInit(), &spec);
  getPredicateVariables(stmt->getCond(), &spec);
  getStride(stmt->getInc(), &spec);
  //now lets examine the body
  addOperations(stmt->getBody(), loop.body);
  //validate that the static analysis succeeded
  validateLoopControlExpr(spec.predicateMax);
  loop.tripCount = getTripCount(&spec);
}

void
ComputeVisitor::visitBodyForStmt(ForStmt* stmt, Loop::Body& body)
{
  checkStmtPragmas(stmt);
  body.subLoops.emplace_back(body.depth+1);
  visitLoop(stmt, body.subLoops.back());
}

void
ComputeVisitor::visitBodyCompoundStmt(CompoundStmt* stmt, Loop::Body& body)
{
  checkStmtPragmas(stmt);
  auto end = stmt->child_end();
  for (auto iter=stmt->child_begin(); iter != end; ++iter){
    addOperations(*iter, body);
  }
}

void
ComputeVisitor::visitBodyCompoundAssignOperator(CompoundAssignOperator* op, Loop::Body& body)
{
  visitBodyBinaryOperator(op, body);
}

void
ComputeVisitor::visitBodyBinaryOperator(BinaryOperator* op, Loop::Body& body)
{
  bool updateLHS = false;
  switch(op->getOpcode()){
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_Assign:
      updateLHS = true; //need to update data flow
    case BO_Mul:
    case BO_Add:
    case BO_Div:
    case BO_Sub:
      if (op->getType()->isIntegerType()) {
        body.intops += 1;
      }
      else if (op->getType()->isFloatingType()) {
        body.flops += 1;
      }
      else if (op->getType()->isPointerType()){
        body.intops += 1;
      }
      else if (op->getType()->isDependentType()){
        //this better be a template type
        const TemplateTypeParmType* ty = cast<const TemplateTypeParmType>(op->getLHS()->getType().getTypePtr());
      }
      else {
        errorAbort(op->getLocStart(), CI,
                   "binary operator in skeletonized loop does not operate on int or float types");
      }
    default:
      break;
  }
  addOperations(op->getLHS(), body, updateLHS);
  addOperations(op->getRHS(), body);
}

void
ComputeVisitor::visitBodyParenExpr(ParenExpr* expr, Loop::Body& body)
{
  addOperations(expr->getSubExpr(), body);
}

void
ComputeVisitor::visitBodyArraySubscriptExpr(ArraySubscriptExpr* expr, Loop::Body& body, bool isLHS)
{
  body.intops += 1; //pointer arithmetic
  //well, well - we might have a unique memory access
  addOperations(expr->getIdx(), body);
  addOperations(expr->getBase(), body);

  //we've added operations - see if this is a new memory access
  MemoryLocation mloc;
  visitAccessArraySubscriptExpr(expr, body, mloc, false); //do not update dependence
  //okay, now we have an array access to check
  AccessHistory& acc = arrays[mloc];
  /**
  std::cout << (isLHS ? "wrote " : "read ")
            << mloc.c_str() << " at rw=("
            << acc.lastReadGeneration << "," << acc.lastWriteGeneration
            << ") during gen " << currentGeneration
            << " with gen dep " << mloc.maxGen
            << " at " << expr->getLocStart().printToString(CI.getSourceManager())
            << std::endl; */

  if (acc.newAccess(mloc.maxGen, currentGeneration, isLHS)){
    TypeInfo ti = CI.getASTContext().getTypeInfo(expr->getType());
    //std::cout << "sizeof(" << expr->getType()->getTypeClassName()
    //          << ")=" << ti.Width << std::endl;

    //fails data flow check OR first access
    if (isLHS) body.writeBytes += ti.Width / 8;
    else       body.readBytes += ti.Width / 8;
  }
}

void
ComputeVisitor::visitBodyDeclRefExpr(DeclRefExpr* expr, Loop::Body& body, bool isLHS)
{
  NamedDecl* decl = expr->getFoundDecl();
  Variable& var = getVariable(decl);
  if (isLHS) var.generation = currentGeneration;
}

void
ComputeVisitor::visitBodyImplicitCastExpr(ImplicitCastExpr* expr, Loop::Body& body, bool isLHS)
{
  addOperations(expr->getSubExpr(), body, isLHS);
}

void
ComputeVisitor::visitBodyCStyleCastExpr(CStyleCastExpr* expr, Loop::Body& body, bool isLHS)
{
  addOperations(expr->getSubExpr(), body, isLHS);
}

void
ComputeVisitor::visitBodyCallExpr(CallExpr* expr, Loop::Body& body)
{
  //the definition of this function better be available
  //procedure calls should NOT be INSIDE a compute intensive loop
  FunctionDecl* callee = expr->getDirectCallee();
  if (callee){
    if (callee->isTemplateInstantiation()){
      FunctionTemplateDecl* td = callee->getPrimaryTemplate();
      addOperations(td->getTemplatedDecl()->getBody(), body);
    } else {
      FunctionDecl* def = callee->getDefinition();
      if (!def){
        std::stringstream sstr;
        sstr << "non-inlineable function " << callee->getNameAsString()
             << " inside compute-intensive code";
        warn(expr->getLocStart(), CI, sstr.str());
      } else {
        addOperations(def->getBody(), body);
      }
    }
  } else {
    warn(expr->getLocStart(), CI,
               "non-inlinable function inside compute-intensive code");
  }
}

void
ComputeVisitor::checkStmtPragmas(Stmt* s)
{
  SSTPragma* prg = pragmas.takeMatch(s);
  if (prg){
    if (prg && prg->cls == SSTPragma::Replace){
      SSTReplacePragma* rprg = static_cast<SSTReplacePragma*>(prg);
      std::list<const Expr*> replaced;
      rprg->run(s, replaced);
      for (auto e : replaced){
        repls.exprs[e] = rprg->replacement();
      }
    }
  }
}

void
ComputeVisitor::visitBodyDeclStmt(DeclStmt* stmt, Loop::Body& body)
{
  SSTPragma* prg = pragmas.takeMatch(stmt);
  if (!stmt->isSingleDecl()){
    return;
  }
  Decl* d = stmt->getSingleDecl();
  if (d){
    if (prg && prg->cls == SSTPragma::Replace){
      SSTReplacePragma* rprg = static_cast<SSTReplacePragma*>(prg);
      repls.decls[d] = rprg->replacement();
    }
    if (isa<VarDecl>(d)){
      VarDecl* vd = cast<VarDecl>(d);
      if (vd->hasInit()){
        addOperations(vd->getInit(), body);
      }
    }
  }
}

void
ComputeVisitor::addOperations(Stmt* stmt, Loop::Body& body, bool isLHS)
{
#define body_case(type,stmt,...) \
  case(clang::Stmt::type##Class): \
    visitBody##type(clang::cast<type>(stmt),__VA_ARGS__); break
  currentGeneration++;
  switch(stmt->getStmtClass()){
    body_case(ForStmt,stmt,body);
    body_case(CompoundStmt,stmt,body);
    body_case(CompoundAssignOperator,stmt,body);
    body_case(BinaryOperator,stmt,body);
    body_case(ParenExpr,stmt,body);
    body_case(CallExpr,stmt,body);
    body_case(DeclRefExpr,stmt,body,isLHS);
    body_case(ImplicitCastExpr,stmt,body,isLHS);
    body_case(CStyleCastExpr,stmt,body,isLHS);
    body_case(ArraySubscriptExpr,stmt,body,isLHS);
    body_case(DeclStmt,stmt,body);
    default:
      break;
  }
#undef body_case
}

void
ComputeVisitor::visitStrideCompoundStmt(clang::CompoundStmt* stmt, ForLoopSpec* spec)
{
  getStride(stmt->body_front(),spec);
}

void
ComputeVisitor::visitStrideCompoundAssignOperator(clang::CompoundAssignOperator* op, ForLoopSpec* spec)
{
  spec->stride = op->getRHS();
}

void
ComputeVisitor::visitStrideUnaryOperator(clang::UnaryOperator* op, ForLoopSpec* spec)
{
  switch(op->getOpcode()){
    case UO_PostInc:
    case UO_PreInc:
      spec->stride = nullptr;
      spec->increment = true;
      break;
    case UO_PostDec:
    case UO_PreDec:
      spec->stride = nullptr;
      spec->increment = false;
      break;
    default:
      errorAbort(op->getLocStart(), CI, "got unary operator that's not a ++/-- increment");
  }
}

#define for_case(feature,type,stmt,spec) \
  case(clang::Stmt::type##Class): \
    visit##feature##type(clang::cast<type>(stmt),spec); break

void
ComputeVisitor::getStride(Stmt* stmt, ForLoopSpec* spec)
{
  switch(stmt->getStmtClass()){
    for_case(Stride,CompoundStmt,stmt,spec);
    for_case(Stride,CompoundAssignOperator,stmt,spec);
    for_case(Stride,UnaryOperator,stmt,spec);
    default:
      errorAbort(stmt->getLocStart(), CI, "bad incrementer expression for skeletonization");
  }
}

void
ComputeVisitor::getPredicateVariables(Expr* expr, ForLoopSpec* spec)
{
  switch(expr->getStmtClass()){
    for_case(Predicate,BinaryOperator,expr,spec);
    default:
      errorAbort(expr->getLocStart(), CI, "skeletonized predicates mut be basic binary ops");
  }
}

void
ComputeVisitor::getInitialVariables(Stmt* stmt, ForLoopSpec* spec)
{
  switch(stmt->getStmtClass()){
    for_case(Initial,DeclStmt,stmt,spec);
    for_case(Initial,BinaryOperator,stmt,spec);
    default:
      errorAbort(stmt->getLocStart(), CI, "bad initial expression for skeletonization");
  }
}
#undef for_case

void
ComputeVisitor::validateLoopControlExpr(Expr* rhs)
{
  //we may have issues if the rhs references variables
  //that are scoped inside the loop
  recurseAll<ValidateScope,NullVisit>(rhs, repls,
                                      CI, scopeStartLine);
}

void
ComputeVisitor::visitPredicateBinaryOperator(BinaryOperator* op, ForLoopSpec* spec)
{
  Expr* rhs = op->getRHS();
  bool keep_going = true;
  while (keep_going){
    switch(rhs->getStmtClass()){
      case Stmt::ImplicitCastExprClass:
        rhs = cast<ImplicitCastExpr>(rhs)->getSubExpr();
        break;
      case Stmt::ParenExprClass:
        rhs = cast<ImplicitCastExpr>(rhs)->getSubExpr();
        break;
      default:
        keep_going = false;
        break;
    }
  }
  spec->predicateMax = rhs;
}

void
ComputeVisitor::visitInitialDeclStmt(clang::DeclStmt* stmt, ForLoopSpec* spec)
{
  if (!stmt->isSingleDecl()){
    errorAbort(stmt->getLocStart(), CI,
               "skeleton for loop initializer is not single declaration");
  }
  VarDecl* var = cast<VarDecl>(stmt->getSingleDecl());
  spec->incrementer = var;
  spec->init = var->getInit();
}

void
ComputeVisitor::visitInitialBinaryOperator(clang::BinaryOperator* op, ForLoopSpec* spec)
{
  if (op->getLHS()->getStmtClass() != Stmt::DeclRefExprClass){
    errorAbort(op->getLocStart(), CI, "skeletonized loop initializer is not simple assignment");
  } else {
    DeclRefExpr* dre = cast<DeclRefExpr>(op->getLHS());
    spec->incrementer = dre->getFoundDecl();
  }
  spec->init = op->getRHS();
}


void 
ComputeVisitor::appendPredicateMax(Expr* max, PrettyPrinter& pp)
{
  auto iter = repls.exprs.find(max);
  if (iter != repls.exprs.end()){
    pp.os << iter->second;
    return;
  }

  //we might have overriden a declaration
  if (max->getStmtClass() == Stmt::DeclRefExprClass){
    DeclRefExpr* dref = cast<DeclRefExpr>(max);
    auto iter = repls.decls.find(dref->getDecl());
    if (iter != repls.decls.end()){
      pp.os << iter->second;
      return;
    }
  }

  pp.print(max);
}

std::string
ComputeVisitor::getTripCount(ForLoopSpec* spec)
{
  PrettyPrinter pp;
  Expr* max = spec->increment ? spec->predicateMax : spec->init;
  Expr* min = spec->increment ? spec->init : spec->predicateMax;

  pp.os << "((";
  if (max) appendPredicateMax(max,pp);
  else pp.os << "0";
  pp.os << ")";
  if (min){
    pp.os << "-";
    pp.os << "(";
    pp.print(min);
    pp.os << ")";
  }
  pp.os << ")";
  if (spec->stride){
    pp.os << "/ (";
    pp.print(spec->stride);
    pp.os << ")";
  }
  return pp.str();
}

void
ComputeVisitor::addLoopContribution(std::ostream& os, Loop& loop)
{
  os << "{ ";
  os << " int tripCount" << loop.body.depth << "=";
  if (loop.body.depth > 0){
    os << "tripCount" << (loop.body.depth-1) << "*";
  }
  os << "(" << loop.tripCount << "); ";
  if (loop.body.flops > 0){
      os << " flops += tripCount" << loop.body.depth << "*" << loop.body.flops << ";";
  }
  if (loop.body.readBytes > 0){
    os << " readBytes += tripCount" << loop.body.depth << "*" << loop.body.readBytes << "; ";
  }
  if (loop.body.writeBytes > 0){
    os << " writeBytes += tripCount" << loop.body.depth << "*" << loop.body.writeBytes << "; ";
  }
  if (loop.body.intops > 0){
      os << " intops += tripCount" << loop.body.depth << "*" << loop.body.intops << ";";
  }

  auto end = loop.body.subLoops.end();
  for (auto iter=loop.body.subLoops.begin(); iter != end; ++iter){
    addLoopContribution(os, *iter);
  }
  os << "}";
}

void
ComputeVisitor::setContext(Stmt* stmt){
  scopeStartLine = stmt->getLocStart();
}

void
ComputeVisitor::replaceStmt(Stmt* stmt, Rewriter& r, Loop& loop)
{
  std::stringstream sstr;
  sstr << "{ uint64_t flops(0); uint64_t readBytes(0); uint64_t writeBytes(0); uint64_t intops(0); ";
  addLoopContribution(sstr, loop);
  sstr << "sstmac_compute_detailed(flops,intops,readBytes); /*assume write-through for now*/";
  sstr << " }";

  PresumedLoc start = CI.getSourceManager().getPresumedLoc(stmt->getLocStart());
  PresumedLoc stop = CI.getSourceManager().getPresumedLoc(stmt->getLocEnd());
  int numLinesDeleted = stop.getLine() - start.getLine();
  //to preserve the correspondence of line numbers
  for (int i=0; i < numLinesDeleted; ++i){
    sstr << "\n";
  }

  r.ReplaceText(stmt->getSourceRange(), sstr.str());
}

