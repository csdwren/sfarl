function checkEq(x,y,msg,tol)

if nargin < 4
  cid = misc.getClass(x);
  if isequal(cid,'single')
    tol = 1e-7; % Tolerance for equality.
  else
    tol = 1e-12;
  end
end
  
assert(gather(abs(x-y) < tol),msg);


  